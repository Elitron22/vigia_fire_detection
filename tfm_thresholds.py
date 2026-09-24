"""Barrido offline de confianza y diagnóstico de errores, solo sobre validación.

Las predicciones se generan una vez a confianza baja usando tfm_evaluation.
Cada punto vuelve a emparejar cajas; no reutiliza etiquetas TP/FP de otro umbral.
No calcula mAP, no entrena y no selecciona una configuración de despliegue.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tfm_evaluation import (
    CLASS_NAMES, box_iou, image_error_record, match_detections, summarize_errors,
)

MODEL_LABELS = {"yolov8s": "YOLOv8s", "yolo26s": "YOLO26s", "yolo26n": "YOLO26n"}
MODEL_COLORS = {"yolov8s": "#275D8C", "yolo26s": "#B86A22", "yolo26n": "#727B38"}
SIZE_LABELS = {"small": "Pequeña (<1 %)", "medium": "Mediana (1–10 %)", "large": "Grande (≥10 %)"}
FP_LABELS = {
    "duplicate": "Duplicada", "wrong_class": "Clase incorrecta",
    "localization": "Localización", "weak_overlap": "Solapamiento débil",
    "no_overlap": "Sin solapamiento",
}


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("split") != "val":
        raise ValueError("El barrido inicial solo admite schema_version=1 y split=val.")
    values = np.asarray(config["thresholds"], dtype=float)
    floor = float(config["prediction_confidence"])
    if not len(values) or not np.isfinite(values).all() or not 0 < floor <= 1:
        raise ValueError("Umbrales no válidos.")
    if np.any(values < floor) or np.any(values > 1) or np.any(np.diff(values) <= 0):
        raise ValueError("La malla debe ser creciente, única y no inferior a la confianza guardada.")
    if config["reference_threshold"] not in values or floor not in values:
        raise ValueError("La malla debe incluir la referencia y el mínimo de inferencia.")
    if config.get("selection_objective") != "macro_recall":
        raise ValueError("Objetivo admitido: macro_recall.")
    if not config["experiments"] or len(set(config["experiments"].values())) != len(config["experiments"]):
        raise ValueError("Experimentos ausentes o duplicados.")
    if any(not 0 < b < 1 for b in config["alarm_budgets"]):
        raise ValueError("Los presupuestos de alarma deben estar entre 0 y 1.")
    if config["review_budget"] not in config["alarm_budgets"]:
        raise ValueError("La revisión debe utilizar un escenario declarado.")
    if not 0 < config["match_iou"] <= 1 or not 0 < config["nms_iou"] <= 1:
        raise ValueError("IoU no válido.")
    bounds = config["size_area_boundaries"]
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] < 1:
        raise ValueError("Bandas de tamaño no válidas.")
    return config


def load_predictions(path, manifest, floor):
    """Valida cobertura exacta, clases, coordenadas, confianza y etiquetas guardadas."""
    if set(manifest["split"]) != {"val"} or manifest["filename"].duplicated().any():
        raise ValueError("Se requiere un manifiesto exclusivo de validación con nombres únicos.")
    expected = manifest.set_index("filename").to_dict("index")
    payloads, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            filename = item["filename"]
            if filename not in expected or filename in seen:
                raise ValueError(f"Imagen inesperada o repetida: {filename}")
            seen.add(filename)
            truth, predictions = item["ground_truth"], item["predictions"]
            record = expected[filename]
            if len(truth) != record["box_count"]:
                raise ValueError(f"Número de etiquetas distinto del manifiesto: {filename}")
            for class_id, name in CLASS_NAMES.items():
                if sum(box["class_id"] == class_id for box in truth) != record[f"{name}_boxes"]:
                    raise ValueError(f"Clases GT distintas del manifiesto: {filename}")
            for box in truth + predictions:
                coords = np.asarray(box["xyxy"], dtype=float)
                if box["class_id"] not in CLASS_NAMES or coords.shape != (4,) or not np.isfinite(coords).all():
                    raise ValueError(f"Caja no válida: {filename}")
                if np.any(coords < -1e-6) or np.any(coords > 1 + 1e-6) or np.any(coords[2:] < coords[:2]):
                    raise ValueError(f"Coordenadas fuera de rango o invertidas: {filename}")
            if any(area_fraction(box) <= 0 for box in truth):
                raise ValueError(f"Anotación GT degenerada: {filename}")
            # Ultralytics puede recortar una predicción al borde y dejar área cero.
            # Se conserva como FP, igual que en la evaluación operativa histórica.
            for box in predictions:
                if not np.isfinite(box["confidence"]) or not floor - 1e-7 <= box["confidence"] <= 1:
                    raise ValueError(f"Confianza incompatible con la caché: {filename}")
            if len(predictions) >= 300:
                raise ValueError(f"Se ha alcanzado max_det en {filename}; revisar antes de barrer.")
            payloads.append(item)
    if seen != set(expected):
        raise ValueError(f"Cobertura incompleta: {len(seen)}/{len(expected)} imágenes.")
    return payloads


def area_fraction(box):
    x1, y1, x2, y2 = box["xyxy"]
    return (x2 - x1) * (y2 - y1)


def size_band(box, boundaries=(.01, .10)):
    area = area_fraction(box)
    return "small" if area < boundaries[0] else "medium" if area < boundaries[1] else "large"


def fp_reason(prediction, truth, match_iou):
    """Clasificación geométrica, no una atribución visual de la causa del error."""
    same = [box_iou(prediction["xyxy"], g["xyxy"]) for g in truth if g["class_id"] == prediction["class_id"]]
    other = [box_iou(prediction["xyxy"], g["xyxy"]) for g in truth if g["class_id"] != prediction["class_id"]]
    if max(same, default=0) >= match_iou:
        return "duplicate"
    if max(other, default=0) >= match_iou:
        return "wrong_class"
    if max(same, default=0) >= .10:
        return "localization"
    if max(other, default=0) >= .10:
        return "weak_overlap"
    return "no_overlap"


def evaluate_threshold(payloads, records, threshold, config, model_key, *, details=False):
    rows, gt_rows, detection_rows = [], [], []
    size_counts = {(cls, band): [0, 0] for cls in CLASS_NAMES.values() for band in SIZE_LABELS}
    for item in payloads:
        truth = item["ground_truth"]
        # Ignorar status y matched_gt_index de la inferencia a confianza mínima.
        predictions = [p for p in item["predictions"] if p["confidence"] >= threshold]
        detections, missed = match_detections(truth, predictions, config["match_iou"])
        row = image_error_record(records[item["filename"]], truth, detections, missed)
        row.update(model_key=model_key, threshold=threshold)
        rows.append(row)
        missed = set(missed)
        for index, gt in enumerate(truth):
            band = size_band(gt, config["size_area_boundaries"])
            counts = size_counts[(CLASS_NAMES[gt["class_id"]], band)]
            counts[0] += 1
            counts[1] += int(index not in missed)
        if details:
            _, missed_at_floor = match_detections(truth, item["predictions"], config["match_iou"])
            missed_at_floor = set(missed_at_floor)
            for index, gt in enumerate(truth):
                gt_rows.append({
                    "model_key": model_key, "threshold": threshold, "filename": item["filename"],
                    "gt_index": index, "class_name": CLASS_NAMES[gt["class_id"]],
                    "area_fraction": area_fraction(gt), "size_band": size_band(gt, config["size_area_boundaries"]),
                    "status": "fn" if index in missed else "tp",
                    "fn_diagnostic": ("persists_at_floor" if index in missed_at_floor else "recovered_at_floor") if index in missed else "detected",
                    **dict(zip(("x1", "y1", "x2", "y2"), gt["xyxy"])),
                })
            for prediction in detections:
                detection_rows.append({
                    "model_key": model_key, "threshold": threshold, "filename": item["filename"],
                    "class_name": CLASS_NAMES[prediction["class_id"]], "confidence": prediction["confidence"],
                    "status": prediction["status"], "is_negative_image": not truth,
                    "zero_area_prediction": area_fraction(prediction) <= 0,
                    "fp_reason": fp_reason(prediction, truth, config["match_iou"]) if prediction["status"] == "fp" else "matched",
                    **dict(zip(("x1", "y1", "x2", "y2"), prediction["xyxy"])),
                })
    images = pd.DataFrame(rows)
    summary = summarize_errors(images)
    metrics = {r["scope"]: r for r in summary["box_metrics"]}
    alarms = {r["scope"]: r for r in summary["negative_image_alarms"]}
    result = {"model_key": model_key, "threshold": threshold, "images": len(images),
              "negative_images": summary["negative_images"],
              "negative_images_with_alarm": alarms["any"]["negative_images_with_alarm"],
              "negative_alarm_rate": alarms["any"]["negative_image_false_alarm_rate"],
              "negative_false_boxes": summary["negative_false_positive_boxes"]["all"]}
    result["zero_area_predictions"] = sum(area_fraction(p) <= 0 for item in payloads
                                            for p in item["predictions"] if p["confidence"] >= threshold)
    for scope in ("smoke", "fire", "all_micro"):
        prefix = "micro" if scope == "all_micro" else scope
        for metric in ("tp", "fp", "fn", "precision", "recall", "f1"):
            result[f"{prefix}_{metric}"] = metrics[scope][metric]
    result["macro_recall"] = (result["smoke_recall"] + result["fire_recall"]) / 2
    result["minimum_class_recall"] = min(result["smoke_recall"], result["fire_recall"])
    for cls in CLASS_NAMES.values():
        result[f"negative_{cls}_alarms"] = alarms[cls]["negative_images_with_alarm"]
    size_rows = [{"model_key": model_key, "threshold": threshold, "class_name": cls,
                  "size_band": band, "gt_boxes": total, "tp": tp, "fn": total - tp,
                  "recall": tp / total if total else np.nan}
                 for (cls, band), (total, tp) in size_counts.items()]
    return result, images, pd.DataFrame(size_rows), pd.DataFrame(gt_rows), pd.DataFrame(detection_rows)


def sweep_model(payloads, manifest, config, model_key):
    if set(manifest["split"]) != {"val"}:
        raise ValueError("El barrido no admite train ni test.")
    records = {r["filename"]: r for r in manifest.to_dict("records")}
    metrics, images, sizes = [], [], []
    for threshold in config["thresholds"]:
        row, per_image, per_size, _, _ = evaluate_threshold(payloads, records, threshold, config, model_key)
        metrics.append(row)
        images.append(per_image)
        sizes.append(per_size)
    return pd.DataFrame(metrics), pd.concat(images, ignore_index=True), pd.concat(sizes, ignore_index=True)


def choose_scenarios(metrics, config):
    """Una confianza común a ambas clases; cada modelo tiene su propio umbral."""
    rows = []
    for model_key, table in metrics.groupby("model_key", sort=False):
        def append(name, eligible, budget=None, objective="macro_recall"):
            if eligible.empty:
                rows.append({"model_key": model_key, "scenario": name, "feasible": False,
                             "alarm_budget": budget, "selection_objective": objective})
                return
            ordering = [objective, "minimum_class_recall", "micro_f1", "negative_alarm_rate", "threshold"]
            ordering = list(dict.fromkeys(ordering))
            winner = eligible.sort_values(ordering, ascending=[c == "negative_alarm_rate" for c in ordering], kind="stable").iloc[0]
            rows.append({**winner.to_dict(), "scenario": name, "feasible": True,
                         "alarm_budget": budget, "selection_objective": objective})
        append("reference_025", table[np.isclose(table.threshold, config["reference_threshold"])])
        append("max_f1", table, objective="micro_f1")
        for budget in config["alarm_budgets"]:
            # Decidir con recuentos sin redondear, no con porcentajes de presentación.
            eligible = table[table.negative_images_with_alarm <= budget * table.negative_images + 1e-12]
            append(f"alarm_{round(budget * 100):02d}pct", eligible, budget)
    return pd.DataFrame(rows)


def verify_baseline(metrics, per_image, historical_summary, historical_images, reference):
    """Reconciliación con la inferencia histórica DIRECTA a confianza 0,25."""
    current = metrics[np.isclose(metrics.threshold, reference)].iloc[0]
    differences = []
    for old in historical_summary["box_metrics"]:
        prefix = "micro" if old["scope"] == "all_micro" else old["scope"]
        for field in ("tp", "fp", "fn"):
            if int(current[f"{prefix}_{field}"]) != old[field]:
                differences.append(f"{prefix}_{field}")
    old_alarm = next(r for r in historical_summary["negative_image_alarms"] if r["scope"] == "any")
    if current.negative_images_with_alarm != old_alarm["negative_images_with_alarm"]:
        differences.append("negative_images_with_alarm")
    selected = per_image[np.isclose(per_image.threshold, reference)].set_index("filename")
    old_images = historical_images.set_index("filename")
    if set(selected.index) != set(old_images.index):
        differences.append("filenames")
    else:
        for cls in CLASS_NAMES.values():
            for field in ("tp", "fp", "fn"):
                col = f"{cls}_{field}"
                if not (selected[col].sort_index() == old_images[col].sort_index()).all():
                    differences.append(f"per_image_{col}")
    return {"status": "passed" if not differences else "failed", "reference_threshold": reference,
            "direct_reference_images": len(old_images), "differences": differences}


def build_figures(metrics, scenarios, sizes, gt_details, detection_details, config, output):
    """Figuras nativas del notebook; PNG para lectura y SVG para la memoria."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MultipleLocator

    output = Path(output) / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11, "figure.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    models = list(config["experiments"])
    markers = dict(zip(models, ("o", "s", "^")))
    styles = dict(zip(models, ("-", "--", "-.")))

    def finish(fig, name, note):
        fig.text(.06, .014, note, fontsize=9, color="#51565A", va="bottom")
        fig.savefig(output / f"{name}.png", dpi=170)
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specifications = [("smoke_recall", "Recall de humo"), ("fire_recall", "Recall de fuego"),
                      ("micro_f1", "F1 micro de cajas"), ("negative_alarm_rate", "Imágenes negativas con alarma")]
    for ax, (metric, title) in zip(axes.flat, specifications):
        for model in models:
            rows = metrics[metrics.model_key == model].sort_values("threshold")
            ax.plot(rows.threshold, rows[metric], color=MODEL_COLORS[model], linestyle=styles[model],
                    marker=markers[model], markersize=4, linewidth=1.8, label=MODEL_LABELS[model])
        ax.axvline(config["reference_threshold"], color="#44484C", linestyle=":", linewidth=1)
        ax.set(title=title, xlabel="Umbral de confianza", xlim=(0, 1), ylim=(0, 1))
        if metric == "negative_alarm_rate":
            ax.set_ylim(0, min(1, max(.06, metrics.negative_alarm_rate.max() * 1.12)))
            ax.yaxis.set_major_locator(MultipleLocator(.05 if metrics.negative_alarm_rate.max() > .1 else .01))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#E1E4E6", linewidth=.7)
    fig.suptitle("D-Fire · Barrido inicial en validación", x=.06, ha="left", fontsize=19)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.52, .948), ncol=3, frameon=False)
    fig.subplots_adjust(top=.865, bottom=.11, left=.075, right=.97, hspace=.42, wspace=.22)
    finish(fig, "01_threshold_curves", "1.721 imágenes · 960 cajas de humo, 1.155 de fuego · 783 negativas · Línea vertical: referencia 0,25.\nUn punto por umbral; las líneas unen los puntos evaluados. IoU de emparejamiento 0,50. Fuente: threshold_metrics.csv.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, cls in zip(axes, ("smoke", "fire")):
        for model in models:
            rows = metrics[metrics.model_key == model].sort_values("threshold")
            ax.plot(rows.negative_alarm_rate, rows[f"{cls}_recall"], color=MODEL_COLORS[model],
                    linestyle=styles[model], marker=markers[model], markersize=4, label=MODEL_LABELS[model])
            candidate = scenarios[(scenarios.model_key == model) & (scenarios.scenario == "alarm_02pct") & scenarios.feasible]
            if not candidate.empty:
                row = candidate.iloc[0]
                ax.scatter(row.negative_alarm_rate, row[f"{cls}_recall"], color=MODEL_COLORS[model],
                           marker="*", s=170, edgecolor="#202428", linewidth=.6, zorder=4)
        for budget in config["alarm_budgets"]:
            ax.axvline(budget, color="#BDC2C6", linestyle=":", linewidth=.9)
        ax.set(xlim=(0, .052), ylim=(0, 1), title=f"Recall de {'humo' if cls == 'smoke' else 'fuego'}",
               xlabel="Imágenes negativas con alguna alarma", ylabel="Cajas reales detectadas")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#E1E4E6", linewidth=.7)
    fig.suptitle("Recall frente a falsas alarmas", x=.06, ha="left", fontsize=18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.53, .93), ncol=3, frameon=False)
    fig.subplots_adjust(top=.79, bottom=.23, left=.075, right=.97, wspace=.24)
    finish(fig, "02_recall_alarm_tradeoff", "Vista limitada al 5 % de negativas con alarma (783 imágenes). La tabla conserva todos los puntos.\nEstrellas: candidatos del escenario ≤2 %, seleccionados por recall medio de ambas clases; no son una decisión de despliegue.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    width = .24
    for ax, cls in zip(axes, ("smoke", "fire")):
        for offset, model in enumerate(models):
            table = sizes[(sizes.model_key == model) & np.isclose(sizes.threshold, config["reference_threshold"]) & (sizes.class_name == cls)].set_index("size_band").reindex(SIZE_LABELS)
            ax.bar(np.arange(3) + (offset - 1) * width, table.recall, width,
                   color=MODEL_COLORS[model], label=MODEL_LABELS[model], hatch=("", "//", "..")[offset], linewidth=.4, edgecolor="white")
        labels = [f"{SIZE_LABELS[band].split(' (')[0]}\n{SIZE_LABELS[band].split(' (')[1][:-1]} · n={int(table.loc[band, 'gt_boxes'])}" for band in SIZE_LABELS]
        ax.set(xticks=np.arange(3), xticklabels=labels, ylim=(0, 1), ylabel="Recall de cajas",
               title="Humo" if cls == "smoke" else "Fuego")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#E1E4E6", linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle("Recall según tamaño de la caja · confianza 0,25", x=.06, ha="left", fontsize=18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.53, .93), ncol=3, frameon=False)
    fig.subplots_adjust(top=.79, bottom=.23, left=.075, right=.97, wspace=.24)
    finish(fig, "03_recall_by_size", "Área relativa = área de la caja / área de la imagen original. Bandas exploratorias propias; no son las categorías COCO.\nMismos denominadores por modelo. No mide extensión física del incendio. Fuente: size_metrics.csv.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, cls in zip(axes, ("smoke", "fire")):
        subset = gt_details[np.isclose(gt_details.threshold, config["reference_threshold"]) & (gt_details.class_name == cls) & (gt_details.status == "fn")]
        counts = subset.groupby(["model_key", "fn_diagnostic"]).size().unstack(fill_value=0).reindex(models, fill_value=0)
        bottom = np.zeros(len(models))
        for diagnostic, label, color, hatch in (("persists_at_floor", "Persiste a 0,01", "#275D8C", ""),
                                               ("recovered_at_floor", "Recuperada al bajar a 0,01", "#D18B46", "//")):
            values = counts.get(diagnostic, pd.Series(0, index=models)).to_numpy()
            ax.bar(np.arange(len(models)), values, bottom=bottom, color=color, label=label, hatch=hatch)
            for i, value in enumerate(values):
                if value:
                    ax.text(i, bottom[i] + value / 2, str(int(value)), ha="center", va="center",
                            color="white" if diagnostic == "persists_at_floor" else "#161A1D", fontsize=11,
                            bbox={"facecolor": color, "edgecolor": "none", "pad": 1})
            bottom += values
        ax.set(xticks=np.arange(len(models)), xticklabels=[MODEL_LABELS[m] for m in models],
               ylabel="Cajas reales no detectadas a 0,25", title="Humo" if cls == "smoke" else "Fuego")
        ax.grid(axis="y", color="#E1E4E6", linewidth=.7)
        ax.set_axisbelow(True)
    shared_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, shared_max)
    fig.suptitle("Qué falsos negativos puede recuperar el umbral", x=.06, ha="left", fontsize=18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.53, .93), ncol=2, frameon=False)
    fig.subplots_adjust(top=.79, bottom=.22, left=.075, right=.97, wspace=.24)
    finish(fig, "04_false_negative_diagnosis", "Comparación por caja anotada entre confianza 0,25 y 0,01, manteniendo clase e IoU ≥0,50.\nLa recuperación puede elevar las falsas alarmas. Persistir a 0,01 no significa que la caja sea imposible de detectar.")


def build_review_galleries(payload_map, records, scenarios, gt_details, detection_details, config, output):
    import io
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    output = Path(output)
    gallery_root = output / "review"
    gallery_root.mkdir(exist_ok=True)
    selection_rows = []
    scenario_name = f"alarm_{round(config['review_budget'] * 100):02d}pct"
    for model_key, payloads in payload_map.items():
        candidate = scenarios[(scenarios.model_key == model_key) & (scenarios.scenario == scenario_name) & scenarios.feasible]
        if candidate.empty:
            continue
        threshold = float(candidate.iloc[0].threshold)
        gt = gt_details[(gt_details.model_key == model_key) & np.isclose(gt_details.threshold, threshold)]
        det = detection_details[(detection_details.model_key == model_key) & np.isclose(detection_details.threshold, threshold)]
        choices = [
            ("negative_alarm", det[det.is_negative_image].sort_values(["confidence", "filename"], ascending=[False, True])),
            ("persistent_fn", gt[(gt.status == "fn") & (gt.fn_diagnostic == "persists_at_floor")].sort_values(["area_fraction", "filename"])),
            ("threshold_fn", gt[(gt.status == "fn") & (gt.fn_diagnostic == "recovered_at_floor")].sort_values(["area_fraction", "filename"])),
        ]
        selected, seen = [], set()
        per_group = max(1, config["gallery_images_per_model"] // len(choices))
        for kind, table in choices:
            filenames = table.filename.drop_duplicates().tolist()
            count = 0
            for filename in filenames:
                if filename in seen:
                    continue
                target = table[table.filename == filename].iloc[0]
                selected.append((filename, kind, int(target.gt_index) if "gt_index" in target.index else None))
                seen.add(filename)
                count += 1
                if count == per_group:
                    break
        if not selected:
            continue
        mapping = {p["filename"]: p for p in payloads}
        kinds = {"negative_alarm": "Alarma en imagen negativa", "persistent_fn": "FN persistente: caja pequeña", "threshold_fn": "FN recuperable al bajar umbral"}
        nrows = (len(selected) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.4 * nrows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (filename, kind, target_index) in zip(axes.flat, selected):
            source_path = Path(records[filename]["image_path"])
            raw = source_path.read_bytes()
            if raw[:2] == b"\xff\xd8" and not raw.endswith(b"\xff\xd9"):
                raw += b"\xff\xd9"
            with Image.open(io.BytesIO(raw)) as source:
                picture = source.convert("RGB")
            width, height = picture.size
            ax.imshow(picture)
            item = mapping[filename]
            pred = [p for p in item["predictions"] if p["confidence"] >= threshold]
            detections, missed = match_detections(item["ground_truth"], pred, config["match_iou"])
            for index, box in enumerate(item["ground_truth"]):
                x1, y1, x2, y2 = box["xyxy"]
                color = "#F0E442" if index in missed else "#FFFFFF"
                ax.add_patch(Rectangle((x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height,
                                       fill=False, edgecolor=color, linewidth=1.4, linestyle="--"))
                if index == target_index:
                    ax.annotate(f"FN {CLASS_NAMES[box['class_id']]}", xy=((x1 + x2) * width / 2, (y1 + y2) * height / 2),
                                xytext=(12, 30), textcoords="offset points", color="#F0E442", fontsize=9,
                                arrowprops={"arrowstyle": "->", "color": "#F0E442"},
                                bbox={"facecolor": "#171B20", "alpha": .8, "pad": 2, "edgecolor": "none"})
            for box in detections:
                x1, y1, x2, y2 = box["xyxy"]
                color = "#56B4E9" if box["status"] == "tp" else "#E69F00"
                ax.add_patch(Rectangle((x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height,
                                       fill=False, edgecolor=color, linewidth=1.6))
                ax.text(x1 * width, y2 * height, f"{'H' if box['class_id'] == 0 else 'F'} {box['confidence']:.2f}",
                        color=color, fontsize=8, va="top", bbox={"facecolor": "#171B20", "alpha": .75, "pad": 1, "edgecolor": "none"})
            ax.set_title(f"{filename} · {kinds[kind]}\nGT={len(item['ground_truth'])} · FP={sum(p['status'] == 'fp' for p in detections)} · FN={len(missed)}", loc="left", fontsize=11, pad=9)
            selection_rows.append({"model_key": model_key, "scenario": scenario_name, "threshold": threshold,
                                   "filename": filename, "selection_reason": kind,
                                   "target_gt_index": target_index,
                                   "source_image_rel": f"data/D-Fire/{records[filename].get('official_split', 'train')}/images/{filename}",
                                   "review_status": "pending_visual_inspection", "observation": "", "hypothesis": ""})
            picture.close()
        fig.suptitle(f"{MODEL_LABELS[model_key]} · Revisión dirigida · confianza {threshold:.2f}", x=.06, ha="left", fontsize=18)
        fig.text(.06, .015, "GT: discontinua blanca; FN: discontinua amarilla. Predicciones: TP azul, FP naranja. H=humo; F=fuego.\nLa flecha señala el FN que motivó la selección. Muestra dirigida, no aleatoria; no estima frecuencias de causas visuales.", fontsize=10)
        fig.subplots_adjust(top=.91, bottom=.075, left=.035, right=.97, hspace=.34, wspace=.10)
        fig.savefig(gallery_root / f"{model_key}_review.png", dpi=160)
        plt.close(fig)
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output / "review_selection.csv", index=False)
    return selection


def write_findings(metrics, scenarios, sizes, gt_details, detection_details, config, output):
    output = Path(output)
    labels = {"reference_025": "Referencia 0,25", "max_f1": "Máximo F1 micro",
              "alarm_01pct": "Alarmas ≤1 %", "alarm_02pct": "Alarmas ≤2 %", "alarm_05pct": "Alarmas ≤5 %"}
    lines = ["# Barrido inicial de umbrales y revisión de errores", "",
             "## Alcance", "",
             f"Se han evaluado {len(config['thresholds'])} umbrales por modelo en las mismas 1.721 imágenes de validación de D-Fire: "
             "960 cajas de humo, 1.155 de fuego y 783 imágenes sin ninguna caja real. Los pesos y las particiones se conservan.", "",
             "Una inferencia a confianza 0,01 por modelo conserva las predicciones. El barrido filtra estas cajas "
             "y vuelve a emparejarlas por clase, confianza descendente e IoU ≥0,50, con cada anotación de un solo uso. "
             "El punto 0,25 se ha reconciliado por imagen con la evaluación directa histórica.", "",
             "## Candidatos por escenario", "",
             "Los límites del 1 %, 2 % y 5 % son escenarios exploratorios. Cada modelo utiliza una confianza "
             "común para humo y fuego. Dentro del límite se maximiza la media del recall de ambas clases; "
             "desempates: mayor recall de la clase peor, F1 micro, menos alarmas y mayor umbral. "
             "No se ha elegido ni exportado una configuración de despliegue.", "",
             "| Modelo | Escenario | Confianza | Recall humo | Recall fuego | P micro | F1 micro | Negativas con alarma |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in scenarios.to_dict("records"):
        if not row["feasible"]:
            lines.append(f"| {MODEL_LABELS.get(row['model_key'], row['model_key'])} | {labels[row['scenario']]} | No factible en la malla | — | — | — | — | — |")
        else:
            lines.append(f"| {MODEL_LABELS[row['model_key']]} | {labels[row['scenario']]} | {row['threshold']:.2f} | "
                         f"{row['smoke_recall']:.2%} | {row['fire_recall']:.2%} | {row['micro_precision']:.2%} | {row['micro_f1']:.2%} | "
                         f"{int(row['negative_images_with_alarm'])}/783 ({row['negative_alarm_rate']:.2%}) |")
    lines += ["", "## Qué indican los errores", "",
              "Los siguientes recuentos se refieren a cajas reales perdidas a confianza 0,25. "
              "Una caja recuperable a 0,01 coincide con una predicción retenida al bajar el umbral; esto no "
              "garantiza que ese umbral sea aceptable por su tasa de falsas alarmas.", "",
              "| Modelo | Clase | FN a 0,25 | Recuperados a 0,01 | Persisten a 0,01 |",
              "|---|---|---:|---:|---:|"]
    fn = gt_details[np.isclose(gt_details.threshold, config["reference_threshold"]) & (gt_details.status == "fn")]
    for model in config["experiments"]:
        for cls in CLASS_NAMES.values():
            rows = fn[(fn.model_key == model) & (fn.class_name == cls)]
            recovered = int((rows.fn_diagnostic == "recovered_at_floor").sum())
            lines.append(f"| {MODEL_LABELS[model]} | {cls} | {len(rows)} | {recovered} | {len(rows) - recovered} |")
    lines += ["", "La figura de tamaño compara el recall a 0,25 para cajas con área relativa <1 %, 1–10 % y ≥10 %. "
              "Son bandas diagnósticas propias, no métricas COCO. Las causas de FP se clasifican geométricamente "
              "(duplicación, clase, localización y solapamiento); 'sin solapamiento' no demuestra por sí solo "
              "que la imagen esté bien anotada ni identifica nubes, reflejos o niebla.", "",
              "Las galerías usan una muestra dirigida: alarmas negativas de mayor confianza, FN persistentes de "
              "menor área y FN recuperables de menor área. Las observaciones visuales, si están disponibles, "
              "se conservan separadas en `visual_review.csv` y `REVISION_VISUAL.md`.", "",
              "## Uso para el siguiente experimento", "",
              "Comparar resolución 640 frente a 768 manteniendo datos y protocolo permite estudiar si se recuperan "
              "las cajas pequeñas persistentes. Los FN recuperables orientan el umbral; los persistentes requieren "
              "examinar resolución, ejemplos y anotaciones. Los resultados presentes no demuestran que 768 vaya a mejorar.", "",
              "## Límites de interpretación", "",
              "- El recall y F1 principales cuentan cajas; no equivalen a la detección de eventos de incendio.",
              "- La tasa de alarma es por imagen negativa; no se convierte en alarmas por hora de vídeo.",
              "- Una semilla por modelo y escenas relacionadas entre particiones limitan la generalización. No se afirma superioridad estadística.",
              "- Los candidatos son los mejores puntos de la malla según la regla declarada, no óptimos continuos.",
              "- No se han calibrado probabilidades. Se ha seleccionado un filtro de confianza de forma exploratoria.",
              "- El mAP estándar sigue en la evaluación del notebook 03; este barrido no lo recalcula.",
              "- Selección y diagnóstico se realizan en val. La evaluación final queda pendiente; el baseline YOLOv8s tiene un test histórico ya conocido.", "",
              "## Artefactos", "",
              "- `threshold_metrics.csv`: una fila por modelo y umbral; TP, FP, FN, P/R/F1 por clase y micro.",
              "- `scenario_candidates.csv`: referencia, máximo F1 y candidatos bajo límites de alarma.",
              "- `image_metrics.csv`: trazabilidad por modelo, umbral e imagen.",
              "- `size_metrics.csv`: denominadores y recall por tamaño.",
              "- `ground_truth_review.csv` y `detection_review.csv`: detalle de cajas en los puntos revisados.",
              "- `baseline_verification.json`: reconciliación con la inferencia histórica a 0,25.",
              "- `run_summary.json`, `config.yaml` y `code/`: entradas, versiones, parámetros, código y hashes.",
              "- `figures/`: gráficos en PNG y SVG; `review/`: galerías de errores.", "",
              "Referencias de implementación: [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/) "
              "y [Ultralytics Val](https://docs.ultralytics.com/modes/val/). El protocolo ejecutado queda fijado por el código y la versión local, no por futuros cambios en esas páginas.", ""]
    (output / "RESUMEN_BARRIDO.md").write_text("\n".join(lines), encoding="utf-8")
