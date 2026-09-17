"""Interpretabilidad reproducible para YOLO: Eigen-CAM multiescala y oclusión."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tfm_evaluation import CLASS_NAMES, box_iou, match_detections


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def eigen_activation_map(activation: np.ndarray) -> np.ndarray:
    """Primera componente espacial; el signo se fija por el extremo dominante."""
    values = np.asarray(activation, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Se esperaba C×H×W, recibido {values.shape}")
    channels, height, width = values.shape
    flat = values.reshape(channels, -1).T
    flat = flat - flat.mean(axis=0, keepdims=True)
    if not np.isfinite(flat).all() or np.allclose(flat, 0):
        return np.zeros((height, width), dtype=np.float32)
    _, _, components = np.linalg.svd(flat, full_matrices=False)
    projection = flat @ components[0]
    if abs(float(projection.min())) > abs(float(projection.max())):
        projection = -projection
    projection = np.maximum(projection, 0).reshape(height, width)
    return minmax(projection)


def unletterbox_map(heatmap: np.ndarray, original_shape: tuple[int, int], imgsz: int) -> np.ndarray:
    """Revierte el letterbox cuadrado usado con rect=False."""
    height, width = map(int, original_shape)
    scale = min(imgsz / width, imgsz / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    pad_x = (imgsz - resized_width) / 2
    pad_y = (imgsz - resized_height) / 2
    square = cv2.resize(np.asarray(heatmap, dtype=np.float32), (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(round(pad_x - .1)), int(round(pad_y - .1))
    crop = square[y0:y0 + resized_height, x0:x0 + resized_width]
    if crop.size == 0:
        raise ValueError("El recorte inverso de letterbox quedó vacío")
    return minmax(cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR))


def multiscale_eigencam(activations: dict[int, np.ndarray], original_shape: tuple[int, int],
                        imgsz: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    maps = {}
    for layer, activation in activations.items():
        if activation.ndim == 4:
            activation = activation[0]
        maps[int(layer)] = unletterbox_map(eigen_activation_map(activation), original_shape, imgsz)
    if not maps:
        raise ValueError("No se capturaron activaciones")
    combined = minmax(np.mean(list(maps.values()), axis=0))
    return combined, maps


def result_predictions(result) -> list[dict]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    return [{"class_id": int(cls), "confidence": float(conf), "xyxy": xyxy.tolist()}
            for cls, conf, xyxy in zip(boxes.cls.detach().cpu().numpy(),
                                       boxes.conf.detach().cpu().numpy(),
                                       boxes.xyxyn.detach().cpu().numpy())]


def target_score(predictions: list[dict], class_id: int, target_box: list[float], min_iou: float) -> float:
    candidates = [(box_iou(prediction["xyxy"], target_box), float(prediction["confidence"]))
                  for prediction in predictions if int(prediction["class_id"]) == int(class_id)]
    if not candidates:
        return 0.0
    overlap, confidence = max(candidates, key=lambda pair: (pair[0], pair[1]))
    return confidence if overlap >= min_iou else 0.0


def grid_boxes(height: int, width: int, grid_size: int) -> list[tuple[int, int, int, int]]:
    y_edges = np.linspace(0, height, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, width, grid_size + 1, dtype=int)
    return [(int(x_edges[col]), int(y_edges[row]), int(x_edges[col + 1]), int(y_edges[row + 1]))
            for row in range(grid_size) for col in range(grid_size)]


def blur_baseline(image: np.ndarray) -> np.ndarray:
    shortest = min(image.shape[:2])
    kernel = max(15, int(shortest // 12) | 1)
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def mask_tiles(image: np.ndarray, blurred: np.ndarray, boxes: list[tuple[int, int, int, int]],
               indices: list[int] | np.ndarray) -> np.ndarray:
    variant = image.copy()
    for index in indices:
        x1, y1, x2, y2 = boxes[int(index)]
        variant[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
    return variant


def predict_arrays(model, arrays: list[np.ndarray], *, imgsz: int, conf: float,
                   batch: int, device=0):
    arguments = {"source": arrays, "imgsz": imgsz, "conf": conf, "batch": batch,
                 "device": device, "verbose": False, "rect": False, "max_det": 300,
                 "save": False, "save_txt": False, "save_conf": False,
                 "quantize": "fp32", "agnostic_nms": False}
    return model.predict(**arguments)


def occlusion_sensitivity(model, image: np.ndarray, class_id: int, target_box: list[float],
                          *, grid_size: int, imgsz: int, conf: float, match_iou: float,
                          batch: int, device=0):
    boxes = grid_boxes(*image.shape[:2], grid_size)
    blurred = blur_baseline(image)
    baseline_result = predict_arrays(model, [image], imgsz=imgsz, conf=conf,
                                     batch=1, device=device)[0]
    baseline = target_score(result_predictions(baseline_result), class_id, target_box, match_iou)
    if baseline <= 0:
        raise RuntimeError("La predicción objetivo no se reprodujo a confianza baja")
    variants = [mask_tiles(image, blurred, boxes, [index]) for index in range(len(boxes))]
    results = predict_arrays(model, variants, imgsz=imgsz, conf=conf, batch=batch, device=device)
    scores = np.asarray([target_score(result_predictions(result), class_id, target_box, match_iou)
                         for result in results], dtype=np.float32)
    drops = (baseline - scores) / baseline
    return {"baseline_score": float(baseline), "scores": scores,
            "relative_drops": drops, "heatmap": drops.reshape(grid_size, grid_size),
            "boxes": boxes, "blurred": blurred}


def deletion_curve(model, image: np.ndarray, class_id: int, target_box: list[float],
                   importance: np.ndarray, baseline_score: float, *, fractions: list[float],
                   repeats: int, seed: int, grid_size: int, imgsz: int, conf: float,
                   match_iou: float, batch: int, device=0) -> pd.DataFrame:
    boxes = grid_boxes(*image.shape[:2], grid_size)
    blurred = blur_baseline(image)
    ordering = np.argsort(-np.asarray(importance).reshape(-1), kind="stable")
    rng = np.random.default_rng(seed)
    specifications = []
    for fraction in fractions:
        count = min(len(boxes), int(np.ceil(float(fraction) * len(boxes))))
        specifications.append(("important", 0, float(fraction), ordering[:count]))
        for repeat in range(repeats):
            indices = rng.choice(len(boxes), size=count, replace=False) if count else np.array([], dtype=int)
            specifications.append(("random", repeat, float(fraction), indices))
    variants = [mask_tiles(image, blurred, boxes, indices) for _, _, _, indices in specifications]
    results = predict_arrays(model, variants, imgsz=imgsz, conf=conf, batch=batch, device=device)
    rows = []
    for (strategy, repeat, fraction, indices), result in zip(specifications, results):
        score = target_score(result_predictions(result), class_id, target_box, match_iou)
        rows.append({"strategy": strategy, "repeat": repeat, "fraction": fraction,
                     "masked_tiles": len(indices), "score": score,
                     "score_retained": score / baseline_score})
    return pd.DataFrame(rows)


def select_cases(predictions_path: Path, manifest: pd.DataFrame, cases: list[list[str]],
                 thresholds: dict[int, float], match_iou: float) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Muestra dirigida y determinista; no estima prevalencias ni frecuencia de causas."""
    records = manifest.set_index("filename").to_dict("index")
    candidates: dict[tuple[str, str], list[dict]] = {tuple(case): [] for case in cases}
    payloads = {}
    with Path(predictions_path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            filename = item["filename"]
            if filename not in records:
                continue
            payloads[filename] = item
            truth = item["ground_truth"]
            deployed = [prediction for prediction in item["predictions"]
                        if prediction["confidence"] >= thresholds[int(prediction["class_id"])]]
            detections, missed = match_detections(truth, deployed, match_iou)
            for class_id, class_name in CLASS_NAMES.items():
                tps = [prediction for prediction in detections
                       if prediction["status"] == "tp" and prediction["class_id"] == class_id]
                if tps:
                    target = max(tps, key=lambda prediction: prediction["confidence"])
                    candidates.get(("true_positive", class_name), []).append({
                        "filename": filename, "score": target["confidence"], "target": target,
                        "gt_box": truth[int(target["matched_gt_index"])]["xyxy"],
                    })
                if not truth:
                    alarms = [prediction for prediction in detections if prediction["class_id"] == class_id]
                    if alarms:
                        target = max(alarms, key=lambda prediction: prediction["confidence"])
                        candidates.get(("negative_false_alarm", class_name), []).append({
                            "filename": filename, "score": target["confidence"], "target": target,
                            "gt_box": None,
                        })
                for gt_index in missed:
                    gt = truth[gt_index]
                    if gt["class_id"] != class_id:
                        continue
                    recovered = [prediction for prediction in item["predictions"]
                                 if prediction["class_id"] == class_id
                                 and prediction["confidence"] < thresholds[class_id]
                                 and box_iou(prediction["xyxy"], gt["xyxy"]) >= match_iou]
                    if recovered:
                        target = max(recovered, key=lambda prediction: prediction["confidence"])
                        candidates.get(("borderline_false_negative", class_name), []).append({
                            "filename": filename, "score": target["confidence"], "target": target,
                            "gt_box": gt["xyxy"],
                        })
    selections, selected_payloads, used = [], {}, set()
    for order, case in enumerate(cases, 1):
        mode, class_name = case
        available = [candidate for candidate in candidates[tuple(case)] if candidate["filename"] not in used]
        if not available:
            raise RuntimeError(f"No hay candidato para {case}")
        # Aciertos: percentil central-alto para evitar seleccionar solo el caso más fácil.
        if mode == "true_positive":
            available = sorted(available, key=lambda item: (item["score"], item["filename"]))
            selected = available[int(round(.70 * (len(available) - 1)))]
        else:
            # Falsa alarma más segura y omisión más próxima al umbral.
            selected = sorted(available, key=lambda item: (item["score"], item["filename"]), reverse=True)[0]
        filename = selected["filename"]
        used.add(filename); selected_payloads[filename] = selected
        class_id = next(index for index, name in CLASS_NAMES.items() if name == class_name)
        selections.append({"order": order, "case": mode, "class_name": class_name,
                           "class_id": class_id, "filename": filename,
                           "category": records[filename]["category"],
                           "cached_target_confidence": float(selected["score"]),
                           "operating_threshold": thresholds[class_id],
                           "image_path": records[filename]["image_path"],
                           "label_path": records[filename]["label_path"],
                           "target_box": json.dumps(selected["target"]["xyxy"]),
                           "gt_box": json.dumps(selected["gt_box"]) if selected["gt_box"] is not None else "",
                           "selection_rule": "directed_predeclared_case_not_frequency_sample"})
    return pd.DataFrame(selections), selected_payloads


def heatmap_mass_inside(heatmap: np.ndarray, target_box: list[float]) -> float:
    height, width = heatmap.shape
    x1, y1, x2, y2 = target_box
    x1, x2 = int(np.floor(x1 * width)), int(np.ceil(x2 * width))
    y1, y2 = int(np.floor(y1 * height)), int(np.ceil(y2 * height))
    positive = np.maximum(heatmap, 0)
    total = float(positive.sum())
    return float(positive[max(0,y1):min(height,y2), max(0,x1):min(width,x2)].sum() / total) if total else 0.0


def pointing_game(heatmap: np.ndarray, target_box: list[float]) -> bool:
    row, col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x, y = (col + .5) / heatmap.shape[1], (row + .5) / heatmap.shape[0]
    x1, y1, x2, y2 = target_box
    return bool(x1 <= x <= x2 and y1 <= y <= y2)
