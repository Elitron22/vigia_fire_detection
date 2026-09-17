"""Evaluación operativa D-Fire, acotada en RAM; nunca entrena ni modifica fuentes.

CLI: python tfm_evaluation.py --manifest .../dataset_manifest.csv --model .../best.pt
Los resultados se escriben en un directorio nuevo y solo se marcan completos al
verificar cobertura y reconciliar cajas e imágenes. No se optimizan umbrales.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
import uuid

import numpy as np
import pandas as pd
import psutil

CLASS_NAMES = {0: "smoke", 1: "fire"}


def model_is_end_to_end(model):
    """Detecta la cabeza one-to-one de Ultralytics sin depender del modelo concreto."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "model", None)
    if layers is None:
        return False
    try:
        head = layers[-1]
    except (IndexError, TypeError):
        return False
    return bool(getattr(head, "end2end", False))


def split_artifact_paths(output, split):
    """Nombres explícitos por partición para impedir mezclar val y test."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Partición no permitida: {split!r}")
    output = Path(output)
    return {
        "analysis": output / f"{split}_error_analysis.csv",
        "predictions": output / f"{split}_predictions.jsonl",
        "detections": output / f"{split}_detection_details.csv",
        "ground_truth": output / f"{split}_ground_truth_details.csv",
        "operating_metrics": output / f"{split}_operating_metrics.csv",
        "negative_alarm_rates": output / f"{split}_negative_image_alarm_rates.csv",
        "negative_images": output / f"{split}_negative_images.csv",
        "summary": output / f"{split}_error_summary.json",
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    """Reemplaza solo el estado propio de esta ejecución, nunca una fuente."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_ground_truth(path, expected_count=None):
    path = Path(path)
    if not path.is_file():
        if expected_count == 0:
            return []
        raise FileNotFoundError(f"Falta una etiqueta no negativa: {path}")
    truth = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: se esperaban cinco campos")
        values = np.asarray([float(v) for v in fields])
        cls, cx, cy, width, height = values
        if not np.isfinite(values).all() or cls not in CLASS_NAMES:
            raise ValueError(f"{path}:{number}: clase/coordenadas inválidas")
        xyxy = np.asarray([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
        if width <= 0 or height <= 0 or xyxy.min() < -1e-6 or xyxy.max() > 1 + 1e-6:
            raise ValueError(f"{path}:{number}: caja inválida en la vista corregida")
        truth.append({"class_id": int(cls), "xyxy": xyxy.tolist()})
    if expected_count is not None and len(truth) != int(expected_count):
        raise ValueError(f"Etiqueta/manifiesto no coinciden: {path}, {len(truth)} != {expected_count}")
    return truth


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def match_detections(truth, predictions, iou_threshold=0.5):
    """Greedy por confianza, misma clase, IoU >= umbral, GT de un solo uso."""
    used = set()
    detections = []
    for prediction in sorted(predictions, key=lambda p: -p["confidence"]):
        candidates = [(i, box_iou(prediction["xyxy"], gt["xyxy"])) for i, gt in enumerate(truth)
                      if i not in used and gt["class_id"] == prediction["class_id"]]
        index, overlap = max(candidates, key=lambda pair: pair[1], default=(None, 0.))
        matched = index is not None and overlap >= iou_threshold
        if matched:
            used.add(index)
        detections.append({**prediction, "status": "tp" if matched else "fp",
                           "matched_gt_index": index if matched else None, "match_iou": overlap if matched else None})
    missed = [i for i in range(len(truth)) if i not in used]
    return detections, missed


def image_error_record(record, truth, detections, missed):
    row = {"image_path": str(record["image_path"]), "label_path": str(record["label_path"]),
           "filename": str(record["filename"]), "category": str(record["category"]),
           "is_negative": len(truth) == 0, "gt_count": len(truth), "pred_count": len(detections)}
    for cls, name in CLASS_NAMES.items():
        row[f"{name}_gt"] = sum(g["class_id"] == cls for g in truth)
        row[f"{name}_pred"] = sum(p["class_id"] == cls for p in detections)
        row[f"{name}_tp"] = sum(p["class_id"] == cls and p["status"] == "tp" for p in detections)
        row[f"{name}_fp"] = sum(p["class_id"] == cls and p["status"] == "fp" for p in detections)
        row[f"{name}_fn"] = sum(truth[i]["class_id"] == cls for i in missed)
    row["total_errors"] = sum(row[f"{c}_{e}"] for c in CLASS_NAMES.values() for e in ("fp", "fn"))
    return row


def summarize_errors(table):
    """Denominador de falsas alarmas: imágenes sin NINGUNA caja real."""
    if table.empty or table["image_path"].duplicated().any():
        raise ValueError("La tabla debe contener exactamente una fila por imagen")
    if table.isna().any().any():
        raise ValueError("Valores ausentes en la tabla de errores")
    for name in CLASS_NAMES.values():
        if not (table[f"{name}_tp"] + table[f"{name}_fp"] == table[f"{name}_pred"]).all():
            raise AssertionError("TP + FP debe ser el número de predicciones")
        if not (table[f"{name}_tp"] + table[f"{name}_fn"] == table[f"{name}_gt"]).all():
            raise AssertionError("TP + FN debe ser el número de cajas reales")
    if not (table.gt_count == table.smoke_gt + table.fire_gt).all():
        raise AssertionError("Recuento GT inconsistente")
    if not (table.pred_count == table.smoke_pred + table.fire_pred).all():
        raise AssertionError("Recuento de predicciones inconsistente")
    negative_mask = table.gt_count == 0
    if not (negative_mask == table.is_negative).all():
        raise AssertionError("Clasificación negativa inconsistente")
    negatives = table.loc[negative_mask]
    negative_count = len(negatives)
    alarm_rows = []
    for scope in ("any", "smoke", "fire", "both"):
        if scope == "any":
            alarm = negatives.pred_count > 0
        elif scope == "both":
            alarm = (negatives.smoke_pred > 0) & (negatives.fire_pred > 0)
        else:
            alarm = negatives[f"{scope}_pred"] > 0
        numerator = int(alarm.sum())
        alarm_rows.append({"scope": scope, "negative_images": negative_count,
                           "negative_images_with_alarm": numerator,
                           "negative_image_false_alarm_rate": numerator / negative_count if negative_count else None})
    metric_rows = []
    for scope in ("smoke", "fire", "all_micro"):
        names = list(CLASS_NAMES.values()) if scope == "all_micro" else [scope]
        counts = {key: int(sum(table[f"{name}_{key}"].sum() for name in names)) for key in ("tp", "fp", "fn")}
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        metric_rows.append({"scope": scope, **counts,
                            "precision": tp / (tp + fp) if tp + fp else None,
                            "recall": tp / (tp + fn) if tp + fn else None,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None})
    negative_boxes = {name: int(negatives[f"{name}_fp"].sum()) for name in CLASS_NAMES.values()}
    return {"images": len(table), "negative_images": negative_count,
            "gt_boxes": int(table.gt_count.sum()), "predicted_boxes": int(table.pred_count.sum()),
            "negative_false_positive_boxes": {**negative_boxes, "all": sum(negative_boxes.values())},
            "negative_image_alarms": alarm_rows, "box_metrics": metric_rows}


def normalized_predictions(prediction):
    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        return []
    coordinates = boxes.xyxyn.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    if not np.isfinite(coordinates).all() or not np.isfinite(confidences).all():
        raise ValueError("El modelo produjo coordenadas/confianzas no finitas")
    result = []
    for cls, confidence, xyxy in zip(classes, confidences, coordinates):
        if cls not in CLASS_NAMES:
            raise ValueError(f"Clase predicha inesperada: {cls}")
        result.append({"class_id": int(cls), "confidence": float(confidence), "xyxy": xyxy.tolist()})
    return result


def release_prediction_buffers(model):
    """No deja referencias a imágenes originales en el Predictor entre lotes."""
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        for name in ("results", "batch", "dataset"):
            if hasattr(predictor, name):
                setattr(predictor, name, None)
    gc.collect()


def iter_bounded_predictions(model, records, *, chunk_size=4, conf=.25, nms_iou=.70,
                             imgsz=640, device=0, end_to_end=None):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size debe ser un entero positivo")
    records = list(records)  # Solo metadatos/rutas, nunca imágenes ni objetos Results.
    end_to_end = model_is_end_to_end(model) if end_to_end is None else bool(end_to_end)
    for start in range(0, len(records), chunk_size):
        chunk = records[start:start + chunk_size]
        paths = [str(r["image_path"]) for r in chunk]
        predict_arguments = {
            "source": paths, "stream": True, "batch": chunk_size, "conf": conf,
            "imgsz": imgsz, "device": device, "verbose": False, "rect": False,
            "quantize": "fp32", "max_det": 300, "agnostic_nms": False,
            "save": False, "save_txt": False, "save_conf": False,
        }
        if not end_to_end:
            predict_arguments["iou"] = nms_iou
        stream = model.predict(**predict_arguments)
        seen = 0
        try:
            for prediction in stream:
                if seen >= len(chunk):
                    raise RuntimeError("Ultralytics devolvió más imágenes de las esperadas")
                record = chunk[seen]
                if os.path.abspath(str(prediction.path)) != os.path.abspath(str(record["image_path"])):
                    raise RuntimeError("Desalineación entre imagen y predicción")
                if tuple(prediction.orig_shape) != (int(record["height"]), int(record["width"])):
                    raise RuntimeError(f"Dimensiones distintas del manifiesto: {prediction.path}")
                payload = normalized_predictions(prediction)
                seen += 1
                del prediction
                yield record, payload
            if seen != len(chunk):
                raise RuntimeError(f"Resultados incompletos: {seen}/{len(chunk)}")
        finally:
            if hasattr(stream, "close"):
                stream.close()
            del stream
            release_prediction_buffers(model)


def preflight_manifest(manifest, inventory_path):
    required = {"split", "image_path", "label_path", "filename", "category", "width", "height", "box_count", "smoke_boxes", "fire_boxes"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(manifest.columns))}")
    if manifest.empty or manifest.image_path.duplicated().any() or manifest.filename.duplicated().any():
        raise ValueError("Manifiesto vacío o con imágenes repetidas")
    inventory = []
    for record in manifest.to_dict("records"):
        image_path, label_path = Path(record["image_path"]), Path(record["label_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Falta la vista derivada: {image_path}. Recréala mediante la auditoría; no se sustituye por etiquetas originales.")
        truth = read_ground_truth(label_path, int(record["box_count"]))
        for cls, name in CLASS_NAMES.items():
            if sum(g["class_id"] == cls for g in truth) != int(record[f"{name}_boxes"]):
                raise ValueError(f"Clases del manifiesto y etiqueta no coinciden: {label_path}")
        expected_category = "smoke+fire" if record["smoke_boxes"] and record["fire_boxes"] else "smoke" if record["smoke_boxes"] else "fire" if record["fire_boxes"] else "background"
        if record["category"] != expected_category:
            raise ValueError(f"Categoría inconsistente: {record['filename']}")
        inventory.append({"image_path": str(image_path), "label_path": str(label_path),
                          "image_sha256": sha256_file(image_path),
                          "label_sha256": sha256_file(label_path) if label_path.is_file() else "missing_negative",
                          "gt_boxes": len(truth)})
    pd.DataFrame(inventory).to_csv(inventory_path, index=False)
    return {"images": len(inventory), "negative_images": int((manifest.box_count == 0).sum()),
            "smoke_gt": int(manifest.smoke_boxes.sum()), "fire_gt": int(manifest.fire_boxes.sum())}


def run_error_analysis(model, manifest, output_parent, *, model_path=None, manifest_path=None,
                       split="test", conf=.25, match_iou=.50, nms_iou=.70, imgsz=640,
                       chunk_size=4, device=0, seed=42, ram_limit_gib=6., expected_images=None,
                       expected_negatives=None):
    """Crea una ejecución nueva. El resumen final no existe si el análisis falla."""
    for name, value in (("conf", conf), ("match_iou", match_iou), ("nms_iou", nms_iou)):
        if not 0 < value <= 1:
            raise ValueError(f"{name} debe pertenecer a (0, 1]")
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    if names != CLASS_NAMES:
        raise ValueError(f"Mapeo de clases incorrecto: {names}")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Partición no permitida: {split!r}")
    selected = manifest.loc[manifest.split == split].copy().reset_index(drop=True)
    if expected_images is not None and len(selected) != expected_images:
        raise ValueError(f"Se esperaban {expected_images} imágenes y hay {len(selected)}")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    output = Path(output_parent) / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "evaluation_code.py")
    process = psutil.Process()
    start = time.monotonic()
    peak_rss = process.memory_info().rss
    completed = 0
    end_to_end = model_is_end_to_end(model)
    metadata = {"status": "preflight", "run_id": run_id, "split": split,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "model_path": str(model_path) if model_path else None,
                "model_sha256": sha256_file(model_path) if model_path else None,
                "manifest_path": str(manifest_path) if manifest_path else None,
                "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
                "evaluation_code_sha256": sha256_file(__file__),
                "protocol": {"confidence": conf, "match_iou": match_iou,
                             "requested_nms_iou": nms_iou,
                             "nms_iou": None if end_to_end else nms_iou,
                             "nms_applied": not end_to_end,
                             "end_to_end": end_to_end,
                             "imgsz": imgsz, "chunk_size": chunk_size, "device": str(device),
                             "rect": False, "half": False, "max_det": 300, "agnostic_nms": False,
                             "seed": seed, "matching": "same_class_confidence_greedy_one_to_one",
                             "negative_definition": "zero_ground_truth_boxes_all_classes",
                             "ram_limit_gib": ram_limit_gib},
                "python": platform.python_version(), "platform": platform.platform()}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import ultralytics
    metadata.update({"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    write_json(output / "run_state.json", metadata)
    print(f"Salidas: {output}", flush=True)
    try:
        counts = preflight_manifest(selected, output / "input_inventory.csv")
        if expected_negatives is not None and counts["negative_images"] != expected_negatives:
            raise ValueError(f"Negativos inesperados: {counts['negative_images']} != {expected_negatives}")
        metadata.update({"status": "running", "input_counts": counts,
                         "input_inventory_sha256": sha256_file(output / "input_inventory.csv")})
        write_json(output / "run_state.json", metadata)
        print(f"Preflight correcto: {counts}", flush=True)
        paths = split_artifact_paths(output, split)
        row_fields = list(image_error_record(selected.iloc[0], [], [], []).keys())
        detection_fields = ["image_path", "filename", "class_id", "confidence", "x1", "y1", "x2", "y2", "status", "matched_gt_index", "match_iou"]
        gt_fields = ["image_path", "filename", "gt_index", "class_id", "x1", "y1", "x2", "y2", "status"]
        with paths["analysis"].open("w", newline="", encoding="utf-8") as row_handle, \
             paths["predictions"].open("w", encoding="utf-8") as prediction_handle, \
             paths["detections"].open("w", newline="", encoding="utf-8") as detection_handle, \
             paths["ground_truth"].open("w", newline="", encoding="utf-8") as gt_handle:
            writer = csv.DictWriter(row_handle, fieldnames=row_fields)
            detection_writer = csv.DictWriter(detection_handle, fieldnames=detection_fields)
            gt_writer = csv.DictWriter(gt_handle, fieldnames=gt_fields)
            for w in (writer, detection_writer, gt_writer):
                w.writeheader()
            stream = iter_bounded_predictions(
                model, selected.to_dict("records"), chunk_size=chunk_size,
                conf=conf, nms_iou=nms_iou, imgsz=imgsz, device=device,
                end_to_end=end_to_end,
            )
            try:
                for record, predictions in stream:
                    truth = read_ground_truth(record["label_path"], int(record["box_count"]))
                    detections, missed = match_detections(truth, predictions, match_iou)
                    row = image_error_record(record, truth, detections, missed)
                    writer.writerow(row)
                    prediction_handle.write(json.dumps({"image_path": record["image_path"], "filename": record["filename"],
                                                        "ground_truth": truth, "predictions": detections,
                                                        "false_negative_gt_indices": missed}, allow_nan=False) + "\n")
                    for p in detections:
                        detection_writer.writerow({"image_path": record["image_path"], "filename": record["filename"],
                                                   **{k: p[k] for k in ("class_id", "confidence", "status", "matched_gt_index", "match_iou")},
                                                   **dict(zip(("x1", "y1", "x2", "y2"), p["xyxy"]))})
                    for index, g in enumerate(truth):
                        gt_writer.writerow({"image_path": record["image_path"], "filename": record["filename"],
                                            "gt_index": index, "class_id": g["class_id"], "status": "fn" if index in missed else "tp",
                                            **dict(zip(("x1", "y1", "x2", "y2"), g["xyxy"]))})
                    completed += 1
                    rss = process.memory_info().rss
                    peak_rss = max(peak_rss, rss)
                    if rss > ram_limit_gib * 1024 ** 3 or psutil.virtual_memory().available < 512 * 1024 ** 2:
                        raise MemoryError("Límite preventivo de RAM: resultados parciales conservados. Reduce chunk_size.")
                    if completed % chunk_size == 0 or completed == len(selected):
                        for handle in (row_handle, prediction_handle, detection_handle, gt_handle):
                            handle.flush()
                    if completed % 100 == 0 or completed == len(selected):
                        write_json(output / "run_state.json", {**metadata, "completed_images": completed,
                                                               "peak_sampled_rss_gib": peak_rss / 1024 ** 3})
                        print(f"{completed:,}/{len(selected):,} imágenes | RSS {rss / 1024 ** 3:.2f} GiB | {time.monotonic() - start:.0f} s", flush=True)
            finally:
                stream.close()
        if completed != len(selected):
            raise AssertionError("Cobertura incompleta")
        table = pd.read_csv(paths["analysis"])
        summary = summarize_errors(table)
        if summary["negative_images"] != counts["negative_images"] or summary["gt_boxes"] != counts["smoke_gt"] + counts["fire_gt"]:
            raise AssertionError("No se reconcilian los totales de entrada y salida")
        pd.DataFrame(summary["box_metrics"]).to_csv(paths["operating_metrics"], index=False)
        pd.DataFrame(summary["negative_image_alarms"]).to_csv(paths["negative_alarm_rates"], index=False)
        table.loc[table.is_negative].to_csv(paths["negative_images"], index=False)
        # ru_maxrss capta picos entre muestreos; psutil mantiene portabilidad en Windows.
        try:
            import resource
            peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024))
        except ImportError:
            pass
        metadata.update({"status": "complete", "completed_images": completed,
                         "elapsed_seconds": time.monotonic() - start, "peak_process_rss_gib": peak_rss / 1024 ** 3,
                         "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None})
        summary = {**metadata, **summary,
                   "artifacts": {name: str(path) for name, path in paths.items()},
                   "limitations": ["Umbrales fijos: no son mAP ni P/R del máximo F1 de Ultralytics.",
                                   "all_micro suma cajas; no es la media entre clases del resumen mAP.",
                                   "Falsa alarma por imagen negativa no equivale a falsas alarmas por hora de vídeo.",
                                   "Las etiquetas se consideran referencia; las escenas cercanas entre splits siguen siendo una limitación.",
                                   "rect=False fija padding cuadrado y evita que el tamaño del lote cambie el preprocesado."]}
        write_json(paths["summary"], summary)
        write_json(output / "run_state.json", metadata)
        return output, table, summary
    except BaseException as exc:
        write_json(output / "run_state.json", {**metadata, "status": "incomplete", "completed_images": completed,
                                               "error": f"{type(exc).__name__}: {exc}"})
        raise


def save_error_gallery(table, predictions_path, output_path, *, count=12, seed=42, mode="hardest"):
    """Mosaico pequeño; usa predicciones guardadas, no vuelve a inferir."""
    from PIL import Image, ImageDraw
    if mode == "negative_alarms":
        selection = table.loc[table.is_negative & (table.pred_count > 0)].sort_values(["pred_count", "filename"], ascending=[False, True]).head(count)
    elif mode == "random":
        selection = table.sample(n=min(count, len(table)), random_state=seed)
    else:
        selection = table.sort_values(["total_errors", "filename"], ascending=[False, True]).head(count)
    if selection.empty:
        return None
    wanted = set(selection.image_path)
    saved = {}
    with Path(predictions_path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["image_path"] in wanted:
                saved[item["image_path"]] = item
    if set(saved) != wanted:
        raise ValueError("Faltan predicciones de la muestra visual")
    columns, cell_w, cell_h = 3, 480, 360
    canvas = Image.new("RGB", (columns * cell_w, math.ceil(len(selection) / columns) * cell_h), "#202020")
    for position, record in enumerate(selection.to_dict("records")):
        with Image.open(record["image_path"]) as source:
            thumbnail = source.convert("RGB")
        thumbnail.thumbnail((cell_w, cell_h - 65))
        draw = ImageDraw.Draw(thumbnail)
        data = saved[record["image_path"]]
        for gt in data["ground_truth"]:
            xyxy = [gt["xyxy"][i] * (thumbnail.width if i % 2 == 0 else thumbnail.height) for i in range(4)]
            draw.rectangle(xyxy, outline="#5beb6b", width=2)
        for p in data["predictions"]:
            xyxy = [p["xyxy"][i] * (thumbnail.width if i % 2 == 0 else thumbnail.height) for i in range(4)]
            draw.rectangle(xyxy, outline="#ff9f43", width=2)
            draw.text((xyxy[0], max(0, xyxy[1] - 12)), f"{CLASS_NAMES[p['class_id']]} {p['confidence']:.2f}", fill="#ff9f43")
        x, y = position % columns * cell_w, position // columns * cell_h
        canvas.paste(thumbnail, (x, y + 60))
        caption = f"{record['filename']} | GT verde, pred naranja\nS FP/FN={record['smoke_fp']}/{record['smoke_fn']} | F FP/FN={record['fire_fp']}/{record['fire_fn']}"
        ImageDraw.Draw(canvas).text((x + 5, y + 5), caption, fill="white")
        thumbnail.close()
    canvas.save(output_path)
    canvas.close()
    return Path(output_path)


def write_summary_markdown(output, summary):
    alarms = summary["negative_image_alarms"][0]
    split = summary["split"]
    nms_value = summary["protocol"]["nms_iou"]
    nms_text = f"{nms_value}" if nms_value is not None else "no aplica (modelo end-to-end)"
    threshold_note = (
        "No se ha reentrenado. Esta evaluación de validación puede utilizarse para "
        "comparar modelos y, más adelante, calibrar el umbral."
        if split == "val"
        else "No se ha reentrenado ni seleccionado un umbral sobre el test."
    )
    rows = [f"# Evaluación operativa D-Fire — {split}", "", threshold_note, "",
            f"- Modelo: `{summary['model_path']}`.",
            f"- Partición: `{split}`.",
            f"- Confianza: {summary['protocol']['confidence']}; IoU de emparejamiento: {summary['protocol']['match_iou']}; NMS IoU: {nms_text}.",
            f"- Cobertura: {summary['images']} imágenes; {summary['negative_images']} negativas.",
            f"- Negativas con alguna alarma: {alarms['negative_images_with_alarm']}/{alarms['negative_images']} ({alarms['negative_image_false_alarm_rate']:.2%}).",
            f"- Cajas falsas sobre imágenes negativas: {summary['negative_false_positive_boxes']['all']}.",
            f"- Pico de RAM del proceso: {summary['peak_process_rss_gib']:.2f} GiB.", "",
            "| Clase | TP | FP | FN | Precisión | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for metric in summary["box_metrics"]:
        values = [f"{metric[k]:.2%}" if metric[k] is not None else "N/D" for k in ("precision", "recall", "f1")]
        rows.append(f"| {metric['scope']} | {metric['tp']} | {metric['fp']} | {metric['fn']} | " + " | ".join(values) + " |")
    rows.extend(["", "## Limitaciones", ""] + [f"- {s}" for s in summary["limitations"]])
    rows.extend(["", f"Los CSV de detalles permiten reconciliar cada caja y cada imagen. `{split}_predictions.jsonl` conserva coordenadas normalizadas, confianza y emparejamientos. Las galerías se generan desde esas predicciones, sin una segunda inferencia.", ""])
    (Path(output) / "RESUMEN.md").write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--match-iou", type=float, default=.50)
    parser.add_argument("--nms-iou", type=float, default=.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ram-limit-gib", type=float, default=6.)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--expected-negatives", type=int)
    args = parser.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    output, table, summary = run_error_analysis(
        model, pd.read_csv(args.manifest), args.output_parent or args.manifest.parent / "error_analysis",
        model_path=args.model, manifest_path=args.manifest, chunk_size=args.chunk_size, device=args.device,
        conf=args.confidence, match_iou=args.match_iou, nms_iou=args.nms_iou, imgsz=args.imgsz,
        split=args.split,
        seed=args.seed, ram_limit_gib=args.ram_limit_gib, expected_images=args.expected_images,
        expected_negatives=args.expected_negatives)
    paths = split_artifact_paths(output, args.split)
    for mode, filename in (("hardest", "hardest_examples.png"), ("negative_alarms", "negative_false_alarms.png"), ("random", "qualitative_predictions.png")):
        save_error_gallery(table, paths["predictions"], output / filename, mode=mode, seed=args.seed)
    write_summary_markdown(output, summary)
    print(json.dumps({"status": "complete", "output": str(output), "negative_image_alarms": summary["negative_image_alarms"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
