"""Ejecuta una sola evaluación final bloqueada del YOLO26s congelado sobre test."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline
from tfm_evaluation import (
    CLASS_NAMES, box_iou, image_error_record, match_detections, model_is_end_to_end,
    run_error_analysis, split_artifact_paths, summarize_errors,
)
from tfm_thresholds import area_fraction, fp_reason, size_band
from tools.freeze_final_model import load_config


def metric_array(metric, name: str) -> np.ndarray:
    value = getattr(metric, name, None)
    if value is None:
        return np.array([], dtype=float)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def safe_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def standard_rows(metrics) -> pd.DataFrame:
    box = metrics.box
    precision = metric_array(box, "p")
    recall = metric_array(box, "r")
    ap50 = metric_array(box, "ap50")
    ap = metric_array(box, "ap")
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        p, r = float(precision[class_id]), float(recall[class_id])
        rows.append({"scope": class_name, "precision": p, "recall": r, "f1": safe_f1(p, r),
                     "mAP50": float(ap50[class_id]), "mAP50_95": float(ap[class_id])})
    p, r = float(precision.mean()), float(recall.mean())
    rows.insert(0, {"scope": "all_macro", "precision": p, "recall": r, "f1": safe_f1(p, r),
                    "mAP50": float(box.map50), "mAP50_95": float(box.map)})
    return pd.DataFrame(rows)


def operational_evaluation(predictions_path: Path, manifest: pd.DataFrame, config: dict, output: Path):
    point = config["operating_point"]
    thresholds = {0: float(point["smoke_threshold"]), 1: float(point["fire_threshold"])}
    match_iou = float(point["match_iou"])
    bounds = point["size_area_boundaries"]
    records = {row["filename"]: row for row in manifest.to_dict("records")}
    rows, gt_rows, detection_rows = [], [], []
    matrix = {(actual, predicted): 0 for actual in ("smoke", "fire", "background")
              for predicted in ("smoke", "fire", "background")}
    seen = set()

    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            filename = item["filename"]
            if filename not in records or filename in seen:
                raise ValueError(f"Predicción inesperada o repetida: {filename}")
            seen.add(filename)
            truth = item["ground_truth"]
            predictions = [p for p in item["predictions"]
                           if float(p["confidence"]) >= thresholds[int(p["class_id"])]]
            detections, missed = match_detections(truth, predictions, match_iou)
            row = image_error_record(records[filename], truth, detections, missed)
            row["max_confidence"] = max((float(p["confidence"]) for p in predictions), default=0.0)
            rows.append(row)

            missed_set = set(missed)
            for index, gt in enumerate(truth):
                cls = CLASS_NAMES[int(gt["class_id"])]
                gt_rows.append({"filename": filename, "gt_index": index, "class_name": cls,
                                "area_fraction": area_fraction(gt),
                                "size_band": size_band(gt, bounds),
                                "status": "fn" if index in missed_set else "tp",
                                **dict(zip(("x1", "y1", "x2", "y2"), gt["xyxy"]))})
            for pred_index, prediction in enumerate(detections):
                cls = CLASS_NAMES[int(prediction["class_id"])]
                detection_rows.append({"filename": filename, "prediction_index": pred_index,
                                       "class_name": cls, "confidence": prediction["confidence"],
                                       "threshold": thresholds[int(prediction["class_id"])],
                                       "status": prediction["status"], "is_negative_image": not truth,
                                       "fp_reason": fp_reason(prediction, truth, match_iou)
                                       if prediction["status"] == "fp" else "matched",
                                       **dict(zip(("x1", "y1", "x2", "y2"), prediction["xyxy"]))})

            used_gt = {int(p["matched_gt_index"]) for p in detections if p["status"] == "tp"}
            for p in detections:
                if p["status"] == "tp":
                    name = CLASS_NAMES[int(p["class_id"])]
                    matrix[(name, name)] += 1
            unmatched_gt = set(range(len(truth))) - used_gt
            unmatched_predictions = [p for p in detections if p["status"] == "fp"]
            consumed_predictions = set()
            for pred_index in sorted(range(len(unmatched_predictions)),
                                     key=lambda i: -float(unmatched_predictions[i]["confidence"])):
                pred = unmatched_predictions[pred_index]
                options = [(gt_index, box_iou(pred["xyxy"], truth[gt_index]["xyxy"]))
                           for gt_index in unmatched_gt
                           if int(truth[gt_index]["class_id"]) != int(pred["class_id"])]
                gt_index, overlap = max(options, key=lambda pair: pair[1], default=(None, 0.0))
                if gt_index is not None and overlap >= match_iou:
                    matrix[(CLASS_NAMES[int(truth[gt_index]["class_id"])],
                            CLASS_NAMES[int(pred["class_id"])])] += 1
                    unmatched_gt.remove(gt_index)
                    consumed_predictions.add(pred_index)
            for gt_index in unmatched_gt:
                matrix[(CLASS_NAMES[int(truth[gt_index]["class_id"])], "background")] += 1
            for pred_index, pred in enumerate(unmatched_predictions):
                if pred_index not in consumed_predictions:
                    matrix[("background", CLASS_NAMES[int(pred["class_id"])])] += 1

    if seen != set(records):
        raise ValueError(f"Cobertura operacional incompleta: {len(seen)}/{len(records)}")
    images = pd.DataFrame(rows)
    gt_details = pd.DataFrame(gt_rows)
    detection_details = pd.DataFrame(detection_rows)
    summary = summarize_errors(images)
    class_metrics = pd.DataFrame(summary["box_metrics"])
    macro_recall = float(class_metrics[class_metrics.scope.isin(["smoke", "fire"])].recall.mean())
    minimum_class_recall = float(class_metrics[class_metrics.scope.isin(["smoke", "fire"])].recall.min())
    global_metrics = class_metrics[class_metrics.scope == "all_micro"].copy()
    global_metrics["macro_recall"] = macro_recall
    global_metrics["minimum_class_recall"] = minimum_class_recall
    global_metrics["smoke_threshold"] = thresholds[0]
    global_metrics["fire_threshold"] = thresholds[1]
    alarms = pd.DataFrame(summary["negative_image_alarms"])
    if gt_details.empty:
        size_metrics = pd.DataFrame(columns=["class_name", "size_band", "gt_boxes", "tp", "fn", "recall"])
    else:
        size_metrics = (gt_details.groupby(["class_name", "size_band"], observed=True)
                        .status.value_counts().unstack(fill_value=0).reset_index())
        for column in ("tp", "fn"):
            if column not in size_metrics:
                size_metrics[column] = 0
        size_metrics["gt_boxes"] = size_metrics.tp + size_metrics.fn
        size_metrics["recall"] = size_metrics.tp / size_metrics.gt_boxes
        order = pd.CategoricalDtype(["small", "medium", "large"], ordered=True)
        size_metrics["size_band"] = size_metrics.size_band.astype(order)
        size_metrics = size_metrics.sort_values(["class_name", "size_band"])
    confusion = pd.DataFrame(0, index=["smoke", "fire", "background"],
                             columns=["smoke", "fire", "background"], dtype=int)
    for (actual, predicted), count in matrix.items():
        confusion.loc[actual, predicted] = count
    confusion.index.name = "actual"
    error_matrix = class_metrics[["scope", "tp", "fp", "fn"]].copy()

    images.to_csv(output / "test_image_metrics.csv", index=False)
    class_metrics.to_csv(output / "test_class_metrics.csv", index=False)
    global_metrics.to_csv(output / "test_operating_metrics.csv", index=False)
    alarms.to_csv(output / "test_negative_image_alarms.csv", index=False)
    size_metrics.to_csv(output / "test_size_metrics.csv", index=False)
    gt_details.to_csv(output / "test_ground_truth_details.csv", index=False)
    detection_details.to_csv(output / "test_detection_details.csv", index=False)
    confusion.to_csv(output / "test_confusion_matrix.csv")
    error_matrix.to_csv(output / "test_error_matrix.csv", index=False)
    return images, class_metrics, global_metrics, alarms, size_metrics, gt_details, detection_details, confusion


def build_comparisons(output: Path, validation: Path, standard: pd.DataFrame,
                      class_metrics: pd.DataFrame, global_metrics: pd.DataFrame,
                      alarms: pd.DataFrame, size_metrics: pd.DataFrame):
    val_standard = pd.read_csv(validation / "standard_metrics.csv")
    val_standard = val_standard[val_standard.candidate == "yolo26s"].copy()
    val_standard["scope"] = val_standard.scope.replace({"all": "all_macro"})
    standard_comparison = val_standard[["scope", "precision", "recall", "mAP50", "mAP50_95"]].merge(
        standard, on="scope", suffixes=("_val", "_test"), validate="one_to_one")
    for metric in ("precision", "recall", "mAP50", "mAP50_95"):
        standard_comparison[f"{metric}_delta_test_minus_val"] = (
            standard_comparison[f"{metric}_test"] - standard_comparison[f"{metric}_val"])
    standard_comparison.to_csv(output / "validation_test_standard_comparison.csv", index=False)

    val_operating = pd.read_csv(validation / "final_operating_points.csv")
    val_operating = val_operating[val_operating.candidate == "yolo26s"].iloc[0]
    test_global = global_metrics.iloc[0]
    alarm_any = alarms[alarms.scope == "any"].iloc[0]
    global_rows = []
    mappings = {
        "micro_precision": (val_operating.micro_precision, test_global.precision),
        "micro_recall": (val_operating.micro_recall, test_global.recall),
        "micro_f1": (val_operating.micro_f1, test_global.f1),
        "macro_recall": (val_operating.macro_recall, test_global.macro_recall),
        "minimum_class_recall": (val_operating.minimum_class_recall, test_global.minimum_class_recall),
        "negative_alarm_rate": (val_operating.negative_alarm_rate,
                                alarm_any.negative_image_false_alarm_rate),
    }
    for metric, (val_value, test_value) in mappings.items():
        global_rows.append({"metric": metric, "validation": float(val_value), "test": float(test_value),
                            "delta_test_minus_validation": float(test_value) - float(val_value)})
    global_comparison = pd.DataFrame(global_rows)
    global_comparison.to_csv(output / "validation_test_operating_comparison.csv", index=False)

    val_class = pd.read_csv(validation / "class_metrics.csv")
    val_class = val_class[val_class.candidate == "yolo26s"]
    test_class = class_metrics[class_metrics.scope.isin(["smoke", "fire"])].rename(columns={"scope": "class_name"})
    class_comparison = val_class[["class_name", "threshold", "precision", "recall", "f1", "tp", "fp", "fn"]].merge(
        test_class[["class_name", "precision", "recall", "f1", "tp", "fp", "fn"]],
        on="class_name", suffixes=("_val", "_test"), validate="one_to_one")
    for metric in ("precision", "recall", "f1"):
        class_comparison[f"{metric}_delta_test_minus_val"] = class_comparison[f"{metric}_test"] - class_comparison[f"{metric}_val"]
    class_comparison.to_csv(output / "validation_test_class_comparison.csv", index=False)

    val_size = pd.read_csv(validation / "size_metrics.csv")
    val_size = val_size[val_size.candidate == "yolo26s"]
    size_comparison = val_size[["class_name", "size_band", "gt_boxes", "recall"]].merge(
        size_metrics[["class_name", "size_band", "gt_boxes", "recall"]],
        on=["class_name", "size_band"], suffixes=("_val", "_test"), validate="one_to_one")
    size_comparison["recall_delta_test_minus_val"] = size_comparison.recall_test - size_comparison.recall_val
    size_comparison.to_csv(output / "validation_test_size_comparison.csv", index=False)
    return standard_comparison, global_comparison, class_comparison, size_comparison


def build_figures(output: Path, standard_comparison: pd.DataFrame, global_comparison: pd.DataFrame,
                  class_comparison: pd.DataFrame, size_comparison: pd.DataFrame,
                  confusion: pd.DataFrame, alarms: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    paths = []
    colors = {"validation": "#7A8793", "test": "#D95F35"}

    subset = global_comparison[global_comparison.metric.isin(["micro_precision", "micro_recall", "micro_f1", "macro_recall"])]
    labels = ["Precisión\nmicro", "Recall\nmicro", "F1\nmicro", "Recall\nmacro"]
    x = np.arange(len(subset)); width = .36
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for idx, split in enumerate(("validation", "test")):
        pos = x + (idx - .5) * width
        values = subset[split].to_numpy()
        ax.bar(pos, values, width, color=colors[split], label=split.capitalize())
        for xp, value in zip(pos, values): ax.text(xp, value + .015, f"{value:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Punto operativo congelado: validación frente a test", loc="left", fontsize=15)
    ax.set_ylabel("Proporción"); ax.grid(axis="y", color="#DFE3E6"); ax.legend(frameon=False)
    fig.tight_layout(); path = figure_dir / "01_operating_validation_vs_test.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, cls in zip(axes, ("smoke", "fire")):
        row = class_comparison[class_comparison.class_name == cls].iloc[0]
        x = np.arange(3)
        for idx, split in enumerate(("val", "test")):
            vals = [row[f"{m}_{split}"] for m in metrics]
            pos = x + (idx - .5) * width
            ax.bar(pos, vals, width, color=colors["validation" if split == "val" else "test"],
                   label="Validación" if split == "val" else "Test")
            for xp, value in zip(pos, vals): ax.text(xp, value + .015, f"{value:.1%}", ha="center", fontsize=9)
        ax.set_xticks(x, ["Precisión", "Recall", "F1"]); ax.set_ylim(0, 1)
        ax.set_title("Humo" if cls == "smoke" else "Fuego")
        ax.grid(axis="y", color="#DFE3E6"); ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Proporción"); axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Rendimiento por clase con umbrales 0,36 / 0,16", fontsize=15)
    fig.tight_layout(); path = figure_dir / "02_class_validation_vs_test.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    bands = ["small", "medium", "large"]
    for ax, cls in zip(axes, ("smoke", "fire")):
        rows = size_comparison[size_comparison.class_name == cls].set_index("size_band").loc[bands]
        x = np.arange(3)
        for idx, split in enumerate(("val", "test")):
            vals = rows[f"recall_{split}"].to_numpy()
            pos = x + (idx - .5) * width
            ax.bar(pos, vals, width, color=colors["validation" if split == "val" else "test"],
                   label="Validación" if split == "val" else "Test")
        ax.set_xticks(x, ["Pequeño", "Mediano", "Grande"]); ax.set_ylim(0, 1)
        ax.set_title("Humo" if cls == "smoke" else "Fuego")
        ax.grid(axis="y", color="#DFE3E6"); ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Recall de cajas"); axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Recall por tamaño: validación frente a test", fontsize=15)
    fig.tight_layout(); path = figure_dir / "03_size_validation_vs_test.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    image = axes[0].imshow(confusion.to_numpy(), cmap="Blues")
    for i in range(3):
        for j in range(3): axes[0].text(j, i, str(int(confusion.iloc[i, j])), ha="center", va="center")
    axes[0].set_xticks(range(3), ["Humo", "Fuego", "Fondo"])
    axes[0].set_yticks(range(3), ["Humo", "Fuego", "Fondo"])
    axes[0].set_xlabel("Predicción"); axes[0].set_ylabel("Real"); axes[0].set_title("Matriz de confusión operativa")
    any_alarm = alarms[alarms.scope == "any"].iloc[0]
    rate = float(any_alarm.negative_image_false_alarm_rate)
    axes[1].bar(["Negativas con alarma"], [rate], color=["#D95F35"], width=.55)
    axes[1].axhline(.01, color="#202428", linestyle="--", label="Límite previo 1 %")
    axes[1].text(0, rate + .0005, f"{rate:.3%}", ha="center", fontsize=11)
    axes[1].set_ylim(0, max(.02, rate * 1.45)); axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    axes[1].set_title(f"Negativas: {int(any_alarm.negative_images_with_alarm)}/{int(any_alarm.negative_images)} con alarma")
    axes[1].legend(frameon=False); axes[1].grid(axis="y", color="#DFE3E6")
    fig.suptitle("Errores y alarmas en test", fontsize=15)
    fig.tight_layout(); path = figure_dir / "04_errors_and_negative_alarms.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def draw_examples(output: Path, predictions_path: Path, manifest: pd.DataFrame,
                  images: pd.DataFrame, detections: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, Path]:
    candidates = []
    used = set()
    def add(reason: str, frame: pd.DataFrame, count: int):
        for filename in frame.filename.tolist():
            if filename not in used and len([r for r in candidates if r[0] == reason]) < count:
                candidates.append((reason, filename)); used.add(filename)
    frame = images.copy()
    frame["fp"] = frame.smoke_fp + frame.fire_fp
    frame["fn"] = frame.smoke_fn + frame.fire_fn
    frame["tp"] = frame.smoke_tp + frame.fire_tp
    add("alarma_negativa", frame[(frame.is_negative) & (frame.pred_count > 0)].sort_values(["max_confidence", "fp"], ascending=False), 3)
    wrong = detections[(detections.status == "fp") & (detections.fp_reason == "wrong_class")]
    add("confusion_clase", frame[frame.filename.isin(wrong.filename)].sort_values(["fp", "fn"], ascending=False), 2)
    add("falso_negativo", frame[(~frame.is_negative) & (frame.fn > 0)].sort_values(["fn", "gt_count"], ascending=False), 3)
    add("fp_en_positiva", frame[(~frame.is_negative) & (frame.fp > 0)].sort_values(["fp", "max_confidence"], ascending=False), 2)
    add("acierto_representativo", frame[(frame.gt_count > 0) & (frame.total_errors == 0)].sort_values(["tp", "gt_count"], ascending=False), 2)

    wanted = {filename for _, filename in candidates}
    payloads = {}
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["filename"] in wanted: payloads[item["filename"]] = item
    records = manifest.set_index("filename").to_dict("index")
    thresholds = {0: float(config["operating_point"]["smoke_threshold"]),
                  1: float(config["operating_point"]["fire_threshold"])}
    match_iou = float(config["operating_point"]["match_iou"])
    gallery = output / "representative_examples"
    gallery.mkdir(exist_ok=True)
    selection_rows, rendered = [], []
    colors = {"tp": "#1296DB", "fp": "#FF8C00", "fn": "#FFD400", "gt": "#52B788"}
    for index, (reason, filename) in enumerate(candidates, 1):
        item = payloads[filename]
        truth = item["ground_truth"]
        predictions = [p for p in item["predictions"] if p["confidence"] >= thresholds[int(p["class_id"])]]
        matched, missed = match_detections(truth, predictions, match_iou)
        picture = Image.open(records[filename]["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(picture)
        width, height = picture.size
        for gt_index, gt in enumerate(truth):
            x1, y1, x2, y2 = gt["xyxy"]
            status = "fn" if gt_index in set(missed) else "gt"
            box = [x1*width, y1*height, x2*width, y2*height]
            draw.rectangle(box, outline=colors[status], width=4 if status == "fn" else 2)
            draw.text((box[0]+3, box[1]+3), f"GT {CLASS_NAMES[gt['class_id']]}{' FN' if status == 'fn' else ''}", fill=colors[status])
        for pred in matched:
            x1, y1, x2, y2 = pred["xyxy"]
            box = [x1*width, y1*height, x2*width, y2*height]
            draw.rectangle(box, outline=colors[pred["status"]], width=3)
            draw.text((box[0]+3, max(0, box[3]-16)), f"{pred['status'].upper()} {CLASS_NAMES[pred['class_id']]} {pred['confidence']:.2f}", fill=colors[pred["status"]])
        target = gallery / f"{index:02d}_{reason}_{Path(filename).stem}.png"
        picture.save(target)
        rendered.append(target)
        row = frame[frame.filename == filename].iloc[0]
        selection_rows.append({"order": index, "selection_reason": reason, "filename": filename,
                               "gt_boxes": int(row.gt_count), "tp": int(row.tp), "fp": int(row.fp), "fn": int(row.fn),
                               "max_confidence": float(row.max_confidence),
                               "image_rel": pipeline.project_relative(target, ROOT),
                               "usage": "post_hoc_reporting_only_not_for_tuning"})
        picture.close()
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output / "representative_examples.csv", index=False)

    thumbs = []
    for path in rendered:
        im = Image.open(path).convert("RGB"); im.thumbnail((420, 280)); thumbs.append((path, im.copy())); im.close()
    columns, cell_w, cell_h = 3, 440, 330
    rows_count = math.ceil(len(thumbs) / columns) if thumbs else 1
    sheet = Image.new("RGB", (columns*cell_w, rows_count*cell_h), "white")
    sheet_draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    reason_map = dict((filename, reason) for reason, filename in candidates)
    for idx, (path, im) in enumerate(thumbs):
        x, y = (idx % columns)*cell_w, (idx // columns)*cell_h
        sheet.paste(im, (x + 10, y + 30))
        filename = candidates[idx][1]
        sheet_draw.text((x + 10, y + 8), f"{idx+1:02d} {reason_map[filename]} · {filename}", fill="black", font=font)
    contact = output / "figures" / "05_representative_examples.png"
    sheet.save(contact); sheet.close()
    return selection, contact


def markdown_table(df: pd.DataFrame, percent_columns=()) -> str:
    display = df.copy()
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value):.2%}" if pd.notna(value) else "—")
    # Evita depender de `tabulate`: el entorno de evaluación debe quedar congelado.
    def cell(value):
        if pd.isna(value): return "—"
        if isinstance(value, (float, np.floating)): return f"{float(value):.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |"
                 for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(output: Path, standard: pd.DataFrame, class_metrics: pd.DataFrame,
                 global_metrics: pd.DataFrame, alarms: pd.DataFrame, size_metrics: pd.DataFrame,
                 confusion: pd.DataFrame, standard_comparison: pd.DataFrame,
                 global_comparison: pd.DataFrame, class_comparison: pd.DataFrame,
                 size_comparison: pd.DataFrame, selection: pd.DataFrame, config: dict) -> Path:
    global_row = global_metrics.iloc[0]
    alarm = alarms[alarms.scope == "any"].iloc[0]
    budget = float(config["operating_point"]["maximum_negative_alarm_rate"])
    budget_status = "cumple" if float(alarm.negative_image_false_alarm_rate) <= budget else "no cumple"
    std = standard.copy(); std["scope"] = std.scope.map({"all_macro": "Global (macro clases)", "smoke": "Humo", "fire": "Fuego"})
    cls = class_metrics[class_metrics.scope.isin(["smoke", "fire"])].copy()
    cls["scope"] = cls.scope.map({"smoke": "Humo", "fire": "Fuego"})
    sizes = size_metrics.copy(); sizes["class_name"] = sizes.class_name.map({"smoke": "Humo", "fire": "Fuego"})
    sizes["size_band"] = sizes.size_band.astype(str).map({"small": "Pequeño", "medium": "Mediano", "large": "Grande"})
    deltas = global_comparison.copy()
    deltas["metric"] = deltas.metric.map({"micro_precision": "Precisión micro", "micro_recall": "Recall micro",
                                            "micro_f1": "F1 micro", "macro_recall": "Recall macro",
                                            "minimum_class_recall": "Recall clase peor", "negative_alarm_rate": "Alarma negativa"})
    matrix = confusion.copy(); matrix.index = ["Humo real", "Fuego real", "Fondo real"]
    matrix.columns = ["Pred. humo", "Pred. fuego", "Pred. fondo"]
    matrix.insert(0, "Real", matrix.index)
    prior_note = "Sí: un baseline YOLOv8s fue consultado en agosto de 2026" if (ROOT / config["known_prior_test_exposure"]).is_file() else "No consta"
    text = f"""# Evaluación final congelada en test

## tl;dr

Se evaluó una única configuración: **YOLO26s 768→768**, checkpoint congelado y
umbrales humo **0,36** / fuego **0,16**. A estos umbrales obtiene en test
**precisión micro {global_row.precision:.2%}**, **recall micro {global_row.recall:.2%}**,
**F1 micro {global_row.f1:.2%}** y **recall macro {global_row.macro_recall:.2%}**.
Activa {int(alarm.negative_images_with_alarm)} de {int(alarm.negative_images)} imágenes
negativas ({alarm.negative_image_false_alarm_rate:.2%}); por tanto, **{budget_status}**
el límite previo del 1 %. Este resultado queda cerrado: no se han buscado umbrales
ni comparado modelos en test y no se modificará la configuración a partir de él.

## Contrato previo a test

- Modelo: YOLO26s, entrenamiento e inferencia a 768.
- Experimento: `{config['model']['experiment_id']}`.
- Umbrales: humo 0,36; fuego 0,16.
- IoU de acierto: 0,50; NMS solicitado: 0,70.
- Límite operativo decidido en validación: ≤1 % de negativas con alarma.
- Población test: {config['expected_test']['images']} imágenes, {config['expected_test']['negative_images']} negativas,
  {config['expected_test']['smoke_boxes']} cajas de humo y {config['expected_test']['fire_boxes']} de fuego.
- Exposición histórica conocida de test: {prior_note}. El YOLO26s congelado no
  se había evaluado previamente en test, pero no debe describirse el conjunto como totalmente virgen.

## Métricas estándar

La precisión y el recall globales son la media entre clases en el punto de operación
interno de Ultralytics; F1 es su media armónica. El mAP usa IoU 0,50 o la media
0,50:0,95 y es independiente de los umbrales por clase elegidos para despliegue.

{markdown_table(std, ['precision','recall','f1','mAP50','mAP50_95'])}

## Punto operativo fijo por clase

{markdown_table(cls[['scope','tp','fp','fn','precision','recall','f1']], ['precision','recall','f1'])}

Global micro: TP={int(global_row.tp)}, FP={int(global_row.fp)}, FN={int(global_row.fn)}.

## Recall por tamaño

Bandas por fracción de área de la imagen: pequeño <1 %, mediano 1–10 % y grande ≥10 %.

{markdown_table(sizes[['class_name','size_band','gt_boxes','tp','fn','recall']], ['recall'])}

## Matriz de errores

Las confusiones cruzadas emparejan una predicción de clase incorrecta con una GT
no detectada de la otra clase cuando IoU≥0,50. El fondo recoge FP restantes y FN
restantes. Es una matriz operacional a 0,36/0,16, no la matriz estándar de mAP.

{markdown_table(matrix)}

## Imágenes negativas

{markdown_table(alarms[['scope','negative_images','negative_images_with_alarm','negative_image_false_alarm_rate']], ['negative_image_false_alarm_rate'])}

La unidad del límite es la **imagen completamente negativa con al menos una caja**,
no el número de cajas FP ni las alarmas por hora de vídeo.

## Validación frente a test

{markdown_table(deltas[['metric','validation','test','delta_test_minus_validation']], ['validation','test','delta_test_minus_validation'])}

### Estándar por clase

{markdown_table(standard_comparison, [c for c in standard_comparison if c == 'f1' or c.startswith(('precision_', 'recall_', 'mAP50_'))])}

### Punto operativo por clase

{markdown_table(class_comparison, [c for c in class_comparison if 'precision' in c or 'recall' in c or 'f1' in c])}

### Recall por tamaño

{markdown_table(size_comparison, ['recall_val','recall_test','recall_delta_test_minus_val'])}

Las diferencias test−validación describen generalización observada; no se usan
como criterio para retocar pesos, resolución o umbrales.

## Ejemplos representativos

Se exportaron {len(selection)} ejemplos dirigidos de aciertos y errores en
`representative_examples/`. Son ilustraciones post hoc, no una muestra aleatoria
ni evidencia para recalibrar el sistema. Azul=TP, naranja=FP, verde=GT detectada,
amarillo=GT omitida.

## Cierre metodológico

La evaluación queda registrada como la única evaluación final del YOLO26s congelado.
El siguiente trabajo del proyecto debe centrarse en documentación, exportación y
benchmark de despliegue con datos externos o de validación; no en volver a consultar
test ni en seleccionar otro modelo a partir de estas cifras.
"""
    target = output / "RESUMEN_EVALUACION_FINAL_TEST.md"
    target.write_text(text, encoding="utf-8")
    return target


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    if not config_path.is_absolute(): config_path = ROOT / config_path
    config = load_config(config_path.resolve())
    parent = ROOT / config["output_parent"]
    parent.mkdir(parents=True, exist_ok=True)
    state_path = parent / "evaluation_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            output = ROOT / state["run_rel"]
            print(f"La evaluación final ya está cerrada; se reutiliza: {output}")
            return output
        raise RuntimeError(f"Ya existe un acceso final a test con estado {state.get('status')!r}; no se repite automáticamente")

    freeze = ROOT / config["freeze_parent"] / "final"
    freeze_manifest = json.loads((freeze / "freeze_manifest.json").read_text(encoding="utf-8"))
    weights = freeze / freeze_manifest["frozen_weights_rel"]
    if pipeline.sha256_file(weights) != config["model"]["source_weights_sha256"]:
        raise RuntimeError("El checkpoint congelado no supera la verificación")
    contract = pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])
    if pipeline.sha256_file(Path(contract["manifest_path"])) != freeze_manifest["dataset_manifest_sha256"]:
        raise RuntimeError("El manifiesto de datos no coincide con el congelado")
    staged = pipeline.stage_prepared_dataset(contract, workers=8)
    full_manifest = pipeline.rebased_manifest(contract, staged)
    manifest = full_manifest[full_manifest.split == "test"].copy().reset_index(drop=True)
    expected = config["expected_test"]
    observed = {"images": len(manifest), "negative_images": int((manifest.box_count == 0).sum()),
                "smoke_boxes": int(manifest.smoke_boxes.sum()), "fire_boxes": int(manifest.fire_boxes.sum())}
    if observed != expected:
        raise ValueError(f"La población test no coincide: {observed} != {expected}")

    config_hash = pipeline.sha256_file(config_path)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + config_hash[:8]
    output = parent / run_id
    output.mkdir(exist_ok=False)
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(freeze / "freeze_manifest.json", output / "freeze_manifest.json")
    code = output / "code"; code.mkdir()
    for path in (Path(__file__), ROOT / "tools" / "verify_final_test_evaluation.py",
                 ROOT / "tfm_evaluation.py", ROOT / "tfm_thresholds.py"):
        if path.exists(): shutil.copy2(path, code / path.name)
    state = {"schema_version": 1, "status": "started", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "run_rel": pipeline.project_relative(output, ROOT), "model": "YOLO26s 768→768",
             "weights_sha256": pipeline.sha256_file(weights), "split": "test",
             "single_final_evaluation": True, "threshold_search_executed": False,
             "models_evaluated": [config["model"]["experiment_id"]]}
    pipeline.write_json_atomic(state_path, state)

    model = None
    try:
        model = YOLO(str(weights))
        pipeline.validate_class_mapping(model.names)
        device = 0 if torch.cuda.is_available() else "cpu"
        runtime_yaml = pipeline.write_runtime_data_yaml(contract, output / "data_runtime.yaml", staged)
        print("[1/5] Métricas estándar en test...", flush=True)
        metrics = model.val(data=str(runtime_yaml), split="test", imgsz=768,
                            batch=int(config["runtime"]["standard_batch"]), device=device,
                            plots=True, project=str(output), name="ultralytics_standard",
                            exist_ok=False, verbose=True)
        standard = standard_rows(metrics)
        standard["preprocess_ms"] = float(metrics.speed.get("preprocess", np.nan))
        standard["inference_ms"] = float(metrics.speed.get("inference", np.nan))
        standard["loss_ms"] = float(metrics.speed.get("loss", np.nan))
        standard["postprocess_ms"] = float(metrics.speed.get("postprocess", np.nan))
        standard.to_csv(output / "test_standard_metrics.csv", index=False)

        print("[2/5] Predicciones operativas congeladas en test...", flush=True)
        cache_output, _, cache_summary = run_error_analysis(
            model, full_manifest, output / "operational_cache", model_path=weights,
            manifest_path=contract["manifest_path"], split="test",
            conf=float(config["operating_point"]["prediction_confidence"]),
            match_iou=float(config["operating_point"]["match_iou"]),
            nms_iou=float(config["operating_point"]["nms_iou"]), imgsz=768,
            chunk_size=int(config["runtime"]["prediction_chunk_size"]), device=device,
            seed=int(config["runtime"]["seed"]), ram_limit_gib=float(config["runtime"]["ram_limit_gib"]),
            expected_images=expected["images"], expected_negatives=expected["negative_images"])
        predictions_path = split_artifact_paths(cache_output, "test")["predictions"]

        print("[3/5] TP/FP/FN, tamaños, errores y alarmas a 0,36/0,16...", flush=True)
        (images, class_metrics, global_metrics, alarms, size_metrics,
         gt_details, detection_details, confusion) = operational_evaluation(
            predictions_path, manifest, config, output)
        standard_comparison, global_comparison, class_comparison, size_comparison = build_comparisons(
            output, ROOT / config["validation_source"], standard, class_metrics,
            global_metrics, alarms, size_metrics)

        print("[4/5] Figuras y ejemplos representativos...", flush=True)
        figures = build_figures(output, standard_comparison, global_comparison,
                                class_comparison, size_comparison, confusion, alarms)
        selection, contact = draw_examples(output, predictions_path, manifest, images,
                                           detection_details, config)
        figures.append(contact)
        report = write_report(output, standard, class_metrics, global_metrics, alarms,
                              size_metrics, confusion, standard_comparison, global_comparison,
                              class_comparison, size_comparison, selection, config)

        print("[5/5] Sellado de resultados...", flush=True)
        run_summary = {
            "schema_version": 1, "status": "complete", "run_id": run_id,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "split": "test", "test_inference_executed": True,
            "single_final_evaluation": True, "threshold_search_executed": False,
            "model_comparison_executed": False,
            "model_label": config["model"]["label"],
            "experiment_id": config["model"]["experiment_id"],
            "weights_sha256": pipeline.sha256_file(weights),
            "trained_imgsz": 768, "inference_imgsz": 768,
            "smoke_threshold": .36, "fire_threshold": .16,
            "maximum_negative_alarm_rate": .01,
            "observed_test_population": observed,
            "operational_cache_rel": pipeline.project_relative(cache_output, ROOT),
            "operational_cache_summary_sha256": pipeline.sha256_file(split_artifact_paths(cache_output, "test")["summary"]),
            "standard_evaluation_rel": pipeline.project_relative(Path(metrics.save_dir), ROOT),
            "report_rel": pipeline.project_relative(report, ROOT),
            "figures": [pipeline.project_relative(path, ROOT) for path in figures],
            "known_prior_test_exposure": freeze_manifest["known_prior_test_exposure"],
            "output_hashes": {pipeline.project_relative(path, output): pipeline.sha256_file(path)
                              for path in output.rglob("*") if path.is_file() and path.name != "run_summary.json"},
        }
        pipeline.write_json_atomic(output / "run_summary.json", run_summary)
        state.update({"status": "complete", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "run_summary_sha256": pipeline.sha256_file(output / "run_summary.json")})
        pipeline.write_json_atomic(state_path, state)
        pipeline.write_json_atomic(parent / "latest.json", {"run_id": run_id,
            "run_rel": pipeline.project_relative(output, ROOT),
            "run_summary_sha256": pipeline.sha256_file(output / "run_summary.json")})
        return output
    except Exception as exc:
        state.update({"status": "failed", "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        pipeline.write_json_atomic(state_path, state)
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def recover_postprocessing(config_path: str | Path) -> Path:
    """Sella una ejecución cuya inferencia acabó; nunca carga YOLO ni lee imágenes test."""
    config_path = Path(config_path)
    if not config_path.is_absolute(): config_path = ROOT / config_path
    config = load_config(config_path.resolve())
    parent = ROOT / config["output_parent"]
    state_path = parent / "evaluation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "failed":
        raise RuntimeError("La recuperación solo admite un estado failed")
    output = ROOT / state["run_rel"]
    required = ["test_standard_metrics.csv", "test_class_metrics.csv", "test_operating_metrics.csv",
                "test_negative_image_alarms.csv", "test_size_metrics.csv", "test_image_metrics.csv",
                "test_detection_details.csv", "test_confusion_matrix.csv",
                "validation_test_standard_comparison.csv", "validation_test_operating_comparison.csv",
                "validation_test_class_comparison.csv", "validation_test_size_comparison.csv",
                "representative_examples.csv"]
    missing = [name for name in required if not (output / name).is_file()]
    caches = list((output / "operational_cache").glob("*/test_error_summary.json"))
    if missing or len(caches) != 1 or not (output / "ultralytics_standard").is_dir():
        raise RuntimeError(f"No es posible recuperar sin inferencia; faltan {missing}, cachés={len(caches)}")
    cache_summary = json.loads(caches[0].read_text(encoding="utf-8"))
    if cache_summary.get("status") != "complete" or cache_summary.get("completed_images") != 4306:
        raise RuntimeError("La caché operacional no está completa")

    standard = pd.read_csv(output / "test_standard_metrics.csv")
    class_metrics = pd.read_csv(output / "test_class_metrics.csv")
    global_metrics = pd.read_csv(output / "test_operating_metrics.csv")
    alarms = pd.read_csv(output / "test_negative_image_alarms.csv")
    size_metrics = pd.read_csv(output / "test_size_metrics.csv")
    confusion = pd.read_csv(output / "test_confusion_matrix.csv", index_col=0)
    standard_comparison = pd.read_csv(output / "validation_test_standard_comparison.csv")
    global_comparison = pd.read_csv(output / "validation_test_operating_comparison.csv")
    class_comparison = pd.read_csv(output / "validation_test_class_comparison.csv")
    size_comparison = pd.read_csv(output / "validation_test_size_comparison.csv")
    selection = pd.read_csv(output / "representative_examples.csv")
    report = write_report(output, standard, class_metrics, global_metrics, alarms,
                          size_metrics, confusion, standard_comparison, global_comparison,
                          class_comparison, size_comparison, selection, config)
    shutil.copy2(Path(__file__), output / "code" / "postprocessing_recovery.py")
    figures = sorted((output / "figures").glob("*.png"))
    freeze_manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    observed = config["expected_test"]
    run_summary = {
        "schema_version": 1, "status": "complete", "run_id": output.name,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split": "test", "test_inference_executed": True, "inference_attempts": 1,
        "single_final_evaluation": True, "threshold_search_executed": False,
        "model_comparison_executed": False, "postprocessing_recovery": True,
        "recovery_reason": "El formateo Markdown requirió tabulate tras completar ambas pasadas; se sustituyó por un renderizador local sin repetir inferencia.",
        "model_label": config["model"]["label"], "experiment_id": config["model"]["experiment_id"],
        "weights_sha256": freeze_manifest["weights_sha256"],
        "trained_imgsz": 768, "inference_imgsz": 768, "smoke_threshold": .36,
        "fire_threshold": .16, "maximum_negative_alarm_rate": .01,
        "observed_test_population": observed,
        "operational_cache_rel": pipeline.project_relative(caches[0].parent, ROOT),
        "operational_cache_summary_sha256": pipeline.sha256_file(caches[0]),
        "standard_evaluation_rel": pipeline.project_relative(output / "ultralytics_standard", ROOT),
        "report_rel": pipeline.project_relative(report, ROOT),
        "figures": [pipeline.project_relative(path, ROOT) for path in figures],
        "known_prior_test_exposure": freeze_manifest["known_prior_test_exposure"],
        "output_hashes": {pipeline.project_relative(path, output): pipeline.sha256_file(path)
                          for path in output.rglob("*") if path.is_file() and path.name != "run_summary.json"},
    }
    pipeline.write_json_atomic(output / "run_summary.json", run_summary)
    original_failure = {key: state.get(key) for key in ("failed_at_utc", "error_type", "error")}
    state.update({"status": "complete", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "inference_attempts": 1, "postprocessing_recovery": True,
                  "original_postprocessing_failure": original_failure,
                  "run_summary_sha256": pipeline.sha256_file(output / "run_summary.json")})
    pipeline.write_json_atomic(state_path, state)
    pipeline.write_json_atomic(parent / "latest.json", {"run_id": output.name,
        "run_rel": pipeline.project_relative(output, ROOT),
        "run_summary_sha256": pipeline.sha256_file(output / "run_summary.json")})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_test_evaluation.yaml")
    parser.add_argument("--recover-postprocessing", action="store_true",
                        help="Finaliza tablas/informe tras un fallo posterior a inferencia; nunca reinfiere.")
    args = parser.parse_args()
    output = recover_postprocessing(args.config) if args.recover_postprocessing else run(args.config)
    print(f"Evaluación final de test: {output}")


if __name__ == "__main__":
    main()
