"""Compara YOLOv8s entrenado a 640, 768 y 1024 en validación D-Fire."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline
import tfm_evaluation as evaluation
from tfm_thresholds import choose_scenarios, evaluate_threshold, load_predictions, sweep_model
from tools.run_threshold_sweep import get_prediction_cache


VARIANT_COLORS = {
    "train640": "#275D8C",
    "train768": "#C58A1B",
    "train1024": "#B86A22",
}
SIZE_LABELS = {"small": "Pequeña\n<1 %", "medium": "Mediana\n1–10 %", "large": "Grande\n≥10 %"}


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("split") != "val":
        raise ValueError("La comparación solo admite schema_version=1 y split=val.")
    variants = config.get("training_variants", {})
    if set(variants) != {"train640", "train768", "train1024"}:
        raise ValueError("Se requieren exactamente train640, train768 y train1024.")
    if sorted(config.get("evaluation_sizes", [])) != [640, 768, 1024]:
        raise ValueError("La comparación factorial requiere evaluación a 640, 768 y 1024.")
    if config.get("comparison_baseline") != "train640":
        raise ValueError("comparison_baseline debe ser train640.")
    if config.get("focus_variant") not in variants:
        raise ValueError("focus_variant debe identificar una variante declarada.")
    if config.get("focus_evaluation_size") not in config["evaluation_sizes"]:
        raise ValueError("focus_evaluation_size debe ser una resolución de evaluación declarada.")
    thresholds = np.asarray(config["thresholds"], dtype=float)
    if (not len(thresholds) or np.any(np.diff(thresholds) <= 0)
            or config["prediction_confidence"] not in thresholds
            or config["reference_threshold"] not in thresholds):
        raise ValueError("Malla de umbrales incompatible.")
    if config["alarm_budgets"] != [config["review_budget"]]:
        raise ValueError("Esta comparación debe utilizar un único presupuesto de alarma.")
    return config


def metric_array(metric, name):
    value = getattr(metric, name, None)
    if value is None:
        return np.array([], dtype=float)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def standard_validation(experiment, contract, staged_dataset, config, eval_imgsz):
    import torch
    import ultralytics
    from ultralytics import YOLO

    weights = Path(experiment["best_model"])
    identity = {
        "experiment_id": experiment["experiment_id"],
        "checkpoint_sha256": pipeline.sha256_file(weights),
        "manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
        "imgsz": int(eval_imgsz),
        "batch": int(config["standard_val_batch"]),
        "split": "val",
        "rect": True,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    cache = (Path(experiment["experiment_root"]) / "evaluation" / "val"
             / "resolution_standard_cache" / sha256_json(identity)[:16])
    summary_path = cache / "summary.json"
    if summary_path.exists():
        saved = pipeline.read_json(summary_path)
        if saved.get("status") == "complete" and saved.get("identity") == identity:
            print(f"{experiment['experiment_id']} @ {eval_imgsz}: reutilizando val estándar", flush=True)
            return saved

    cache.mkdir(parents=True, exist_ok=True)
    runtime_yaml = pipeline.write_runtime_data_yaml(
        contract, cache / "data_runtime.yaml", staged_dataset
    )
    run_name = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = YOLO(str(weights))
    pipeline.validate_class_mapping(model.names)
    if not torch.cuda.is_available():
        raise RuntimeError("La validación estándar requiere CUDA en este proyecto.")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = model.val(
        data=str(runtime_yaml), split="val", imgsz=int(eval_imgsz),
        batch=int(config["standard_val_batch"]), device=0, plots=False,
        project=str(cache / "runs"), name=run_name, exist_ok=False,
        verbose=False,
    )
    precision = metric_array(result.box, "p")
    recall = metric_array(result.box, "r")
    ap50 = metric_array(result.box, "ap50")
    ap = metric_array(result.box, "ap")
    rows = [{
        "scope": "all", "precision": float(precision.mean()),
        "recall": float(recall.mean()), "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
    }]
    for class_id, class_name in pipeline.CLASS_NAMES.items():
        rows.append({
            "scope": class_name, "precision": float(precision[class_id]),
            "recall": float(recall[class_id]), "mAP50": float(ap50[class_id]),
            "mAP50_95": float(ap[class_id]),
        })
    saved = {
        "status": "complete", "identity": identity, "metrics": rows,
        "speed_ms_per_image": {k: float(v) for k, v in result.speed.items()},
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "save_dir": str(result.save_dir),
    }
    pipeline.write_json_atomic(summary_path, saved)
    del result, model
    gc.collect()
    torch.cuda.empty_cache()
    return saved


def threshold_config(config, eval_imgsz):
    return {
        **config,
        "imgsz": int(eval_imgsz),
        "chunk_size": int(config["prediction_chunk_size"]),
        "ram_limit_gib": float(config["ram_limit_gib"]),
    }


def image_detection_metrics(table):
    rows = {}
    for class_name in ("smoke", "fire"):
        positive = table[f"{class_name}_gt"] > 0
        detected = table[f"{class_name}_tp"] > 0
        rows[f"{class_name}_positive_images"] = int(positive.sum())
        rows[f"{class_name}_images_detected"] = int((positive & detected).sum())
        rows[f"{class_name}_image_recall"] = float(detected[positive].mean())
    positive = table.gt_count > 0
    detected = (table.smoke_tp + table.fire_tp) > 0
    rows["positive_images"] = int(positive.sum())
    rows["positive_images_detected"] = int((positive & detected).sum())
    rows["incident_image_recall"] = float(detected[positive].mean())
    return rows


def error_review_tables(selected_images, config):
    """Resume errores por configuración y selecciona casos del modelo a 768."""
    summary_rows = []
    for key, table in selected_images.items():
        negative = table.gt_count == 0
        positive = ~negative
        fire_positive = table.fire_gt > 0
        fp_total = table.smoke_fp + table.fire_fp
        summary_rows.append({
            "configuration": key,
            "threshold": float(table.threshold.iloc[0]),
            "smoke_false_negatives": int(table.smoke_fn.sum()),
            "fire_false_negatives": int(table.fire_fn.sum()),
            "smoke_false_positives": int(table.smoke_fp.sum()),
            "fire_false_positives": int(table.fire_fp.sum()),
            "false_positive_boxes_in_negative_images": int(fp_total[negative].sum()),
            "false_positive_boxes_in_positive_images": int(fp_total[positive].sum()),
            "negative_images_with_alarm": int((negative & (fp_total > 0)).sum()),
            "positive_images_with_false_positive": int((positive & (fp_total > 0)).sum()),
            "fire_images_missed": int((fire_positive & (table.fire_tp == 0)).sum()),
            "incident_images_missed": int((positive & ((table.smoke_tp + table.fire_tp) == 0)).sum()),
        })

    focus_variant = config["focus_variant"]
    focus_eval_size = int(config["focus_evaluation_size"])
    focus_key = f"{focus_variant}_eval{focus_eval_size}"
    focus = selected_images[focus_key].copy()
    focus["false_positive_boxes"] = focus.smoke_fp + focus.fire_fp
    focus["false_negative_boxes"] = focus.smoke_fn + focus.fire_fn
    focus["error_priority"] = focus.false_positive_boxes + focus.false_negative_boxes
    examples = focus.sort_values(
        ["error_priority", "fire_fn", "false_positive_boxes", "filename"],
        ascending=[False, False, False, True],
    ).head(25)
    columns = [
        "filename", "category", "gt_count", "pred_count", "smoke_gt", "fire_gt",
        "smoke_tp", "fire_tp", "smoke_fp", "fire_fp", "smoke_fn", "fire_fn",
        "false_positive_boxes", "false_negative_boxes", "error_priority",
    ]
    return pd.DataFrame(summary_rows), examples[columns]


def exact_mcnemar(a, b):
    """Prueba binomial exacta para dos resultados booleanos emparejados."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    gains = int((~a & b).sum())
    losses = int((a & ~b).sum())
    discordant = gains + losses
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    return {"gains": gains, "losses": losses, "discordant": discordant, "p_value": p_value}


def paired_native_comparison(native_images, native_gt, baseline_key, candidate_key):
    old = native_images[baseline_key].set_index("filename")
    new = native_images[candidate_key].set_index("filename")
    if not old.index.equals(new.index):
        old, new = old.align(new, join="inner", axis=0)
    fire_positive = old.fire_gt > 0
    fire_image = exact_mcnemar(
        old.loc[fire_positive, "fire_tp"] > 0,
        new.loc[fire_positive, "fire_tp"] > 0,
    )
    fire_image.update({
        "baseline_key": baseline_key,
        "candidate_key": candidate_key,
        "unit": "fire_positive_image",
        "n": int(fire_positive.sum()),
        "baseline_rate": float((old.loc[fire_positive, "fire_tp"] > 0).mean()),
        "candidate_rate": float((new.loc[fire_positive, "fire_tp"] > 0).mean()),
    })

    old_gt = native_gt[baseline_key]
    new_gt = native_gt[candidate_key]
    keys = ["filename", "gt_index", "class_name", "area_fraction", "size_band"]
    paired = old_gt[keys + ["status"]].merge(
        new_gt[keys + ["status"]], on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one"
    )
    paired = paired[paired.class_name == "fire"].copy()
    fire_box = exact_mcnemar(paired.status_baseline == "tp", paired.status_candidate == "tp")
    fire_box.update({
        "baseline_key": baseline_key,
        "candidate_key": candidate_key,
        "unit": "fire_box",
        "n": len(paired),
        "baseline_rate": float((paired.status_baseline == "tp").mean()),
        "candidate_rate": float((paired.status_candidate == "tp").mean()),
    })
    paired["baseline_key"] = baseline_key
    paired["candidate_key"] = candidate_key
    return pd.DataFrame([fire_image, fire_box]), paired


def training_resources(experiments, config):
    rows = []
    for variant, experiment in experiments.items():
        results = pd.read_csv(Path(experiment["training_run_dir"]) / "results.csv")
        best_index = int(results["metrics/mAP50-95(B)"].idxmax())
        rows.append({
            "variant": variant,
            "experiment_id": experiment["experiment_id"],
            "trained_imgsz": int(config["training_variants"][variant]["trained_imgsz"]),
            "epochs_completed": int(results.epoch.iloc[-1]),
            "best_epoch": int(results.epoch.iloc[best_index]),
            "training_seconds": float(results.time.iloc[-1]),
            "training_hours": float(results.time.iloc[-1] / 3600),
            "best_training_map50_95": float(results["metrics/mAP50-95(B)"].iloc[best_index]),
        })
    table = pd.DataFrame(rows)
    baseline = float(table.loc[table.variant == "train640", "training_seconds"].iloc[0])
    table["training_time_ratio_vs_640"] = table.training_seconds / baseline
    return table


def draw_gallery(payloads, thresholds, paired_fire, records, output, baseline_key, candidate_key, count=6):
    from PIL import Image, ImageDraw, ImageFont

    old_payload = {x["filename"]: x for x in payloads[baseline_key]}
    new_payload = {x["filename"]: x for x in payloads[candidate_key]}
    changed = paired_fire.assign(
        change=np.select(
            [(paired_fire.status_baseline == "fn") & (paired_fire.status_candidate == "tp"),
             (paired_fire.status_baseline == "tp") & (paired_fire.status_candidate == "fn")],
            ["recuperada", "perdida"], default="igual"
        )
    )
    selected = pd.concat([
        changed[changed.change == "recuperada"].sort_values(["area_fraction", "filename"]).head((count + 1) // 2),
        changed[changed.change == "perdida"].sort_values(["area_fraction", "filename"]).head(count // 2),
    ], ignore_index=True).drop_duplicates("filename").head(count)
    if selected.empty:
        return pd.DataFrame()

    panels = []
    review_rows = []
    font = ImageFont.load_default()
    for row in selected.itertuples():
        source = Image.open(records[row.filename]["image_path"]).convert("RGB")
        def configuration_label(key):
            variant, eval_size = key.rsplit("_eval", 1)
            return f"{variant.removeprefix('train')} → {eval_size}"

        for key, label in ((baseline_key, configuration_label(baseline_key)),
                           (candidate_key, configuration_label(candidate_key))):
            image = source.copy()
            draw = ImageDraw.Draw(image)
            truth = old_payload[row.filename]["ground_truth"]
            raw_predictions = old_payload[row.filename]["predictions"] if key == baseline_key else new_payload[row.filename]["predictions"]
            threshold = thresholds[key]
            predictions, _ = evaluation.match_detections(
                truth, [p for p in raw_predictions if p["confidence"] >= threshold], 0.50
            )
            width, height = image.size
            for gt in truth:
                x1, y1, x2, y2 = gt["xyxy"]
                draw.rectangle((x1*width, y1*height, x2*width, y2*height), outline="white", width=max(2, width//500))
            for pred in predictions:
                if pred["confidence"] < threshold:
                    continue
                color = "#2F80ED" if pred.get("status") == "tp" else "#F2994A"
                x1, y1, x2, y2 = pred["xyxy"]
                draw.rectangle((x1*width, y1*height, x2*width, y2*height), outline=color, width=max(2, width//500))
            header = Image.new("RGB", (width, 30), "#202428")
            ImageDraw.Draw(header).text((8, 8), f"{label} · umbral {threshold:.2f}", fill="white", font=font)
            panel = Image.new("RGB", (width, height + 30), "white")
            panel.paste(header, (0, 0)); panel.paste(image, (0, 30))
            panels.append((row.filename, row.change, panel))
        review_rows.append({
            "filename": row.filename, "gt_index": int(row.gt_index),
            "area_fraction": float(row.area_fraction), "size_band": row.size_band,
            "change": row.change, "status_baseline": row.status_baseline,
            "status_candidate": row.status_candidate,
            "baseline_key": baseline_key, "candidate_key": candidate_key,
        })

    target_width = 620
    rendered = []
    for filename in selected.filename:
        pair = [p for name, _, p in panels if name == filename]
        resized = []
        for panel in pair:
            ratio = target_width / panel.width
            resized.append(panel.resize((target_width, round(panel.height * ratio))))
        row_height = max(p.height for p in resized)
        canvas = Image.new("RGB", (target_width * 2, row_height + 28), "white")
        for index, panel in enumerate(resized):
            canvas.paste(panel, (index * target_width, 28))
        ImageDraw.Draw(canvas).text((8, 8), f"{filename} · {selected.loc[selected.filename == filename, 'change'].iloc[0]}", fill="#202428", font=font)
        rendered.append(canvas)
    full = Image.new("RGB", (target_width * 2, sum(x.height for x in rendered)), "white")
    y = 0
    for row_image in rendered:
        full.paste(row_image, (0, y)); y += row_image.height
    path = Path(output) / "figures" / "05_changed_fire_examples.png"
    full.save(path, quality=92)
    return pd.DataFrame(review_rows)


def build_figures(metrics, scenarios, standard, size_metrics, image_detection, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figures = Path(output) / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none",
    })
    ordered_variants = sorted(
        config["training_variants"],
        key=lambda variant: int(config["training_variants"][variant]["trained_imgsz"]),
    )
    native = [
        f"{variant}_eval{int(config['training_variants'][variant]['trained_imgsz'])}"
        for variant in ordered_variants
    ]
    labels = [
        f"{int(config['training_variants'][variant]['trained_imgsz'])}\n→ "
        f"{int(config['training_variants'][variant]['trained_imgsz'])}"
        for variant in ordered_variants
    ]
    colors = [VARIANT_COLORS[variant] for variant in ordered_variants]

    selected = scenarios[(scenarios.scenario == "alarm_02pct") & scenarios.feasible].set_index("model_key")
    image_selected = image_detection.set_index("configuration")
    measures = [
        ("Recall de cajas de fuego", [selected.loc[k, "fire_recall"] for k in native]),
        ("Recall de imágenes con fuego", [image_selected.loc[k, "fire_image_recall"] for k in native]),
        ("Precisión micro de cajas", [selected.loc[k, "micro_precision"] for k in native]),
        ("Negativas con alarma", [selected.loc[k, "negative_alarm_rate"] for k in native]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    for panel_index, (ax, (title, values)) in enumerate(zip(axes, measures)):
        bars = ax.bar(labels, values, color=colors, edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3)
        upper = .025 if panel_index == 3 else 1
        ax.set(title=title, ylim=(0, upper)); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="#E1E4E6"); ax.set_axisbelow(True)
    fig.suptitle("Comparación operativa con ≤2 % de falsas alarmas", x=.055, ha="left", fontsize=18)
    fig.subplots_adjust(top=.78, bottom=.22, left=.055, right=.98, wspace=.32)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"01_native_operating_comparison.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)

    native_metrics = metrics[metrics.model_key.isin(native)]
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    markers = ("o", "D", "s", "^")
    for key, label, color, marker in zip(native, labels, colors, markers):
        table = native_metrics[native_metrics.model_key == key].sort_values("threshold")
        mask = table.negative_alarm_rate <= .06
        ax.plot(table.loc[mask, "negative_alarm_rate"], table.loc[mask, "fire_recall"],
                color=color, marker=marker, label=label, linewidth=2)
        point = selected.loc[key]
        ax.scatter(point.negative_alarm_rate, point.fire_recall, color=color, marker="*", s=180,
                   edgecolor="#202428", zorder=4)
    ax.axvline(config["review_budget"], color="#55595C", linestyle=":", label="Límite 2 %")
    ax.set(xlabel="Imágenes negativas con alguna alarma", ylabel="Recall de cajas de fuego",
           title="Recall de fuego frente a falsas alarmas", xlim=(0, .06), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(color="#E1E4E6"); ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figures / f"02_fire_tradeoff.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)

    std_fire = standard[standard.scope == "fire"].pivot(index="trained_imgsz", columns="eval_imgsz", values="mAP50_95").sort_index().sort_index(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    image_matrix = image_detection.pivot(index="trained_imgsz", columns="eval_imgsz", values="fire_image_recall").sort_index().sort_index(axis=1)
    for ax, matrix, title in ((axes[0], std_fire, "mAP50-95 de fuego"), (axes[1], image_matrix, "Recall de imágenes con fuego")):
        ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.1%}", ha="center", va="center",
                        color="white" if matrix.iloc[i, j] > .55 else "#202428", fontsize=13, fontweight="bold")
        ax.set(xticks=range(matrix.shape[1]), xticklabels=matrix.columns,
               yticks=range(matrix.shape[0]), yticklabels=matrix.index,
               xlabel="Resolución de evaluación", ylabel="Resolución de entrenamiento", title=title)
    fig.suptitle("Separación del efecto de entrenamiento e inferencia", x=.07, ha="left", fontsize=17)
    fig.subplots_adjust(top=.78, bottom=.17, left=.11, right=.96, wspace=.35)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"03_factorial_resolution.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)

    size_selected = size_metrics[
        (size_metrics.class_name == "fire")
        & size_metrics.model_key.isin(native)
    ].copy()
    chosen_thresholds = selected["threshold"].to_dict()
    size_selected = size_selected[
        size_selected.apply(
            lambda row: round(float(row.threshold), 6)
            == round(float(chosen_thresholds[row.model_key]), 6),
            axis=1,
        )
    ]
    size_order = ["small", "medium", "large"]
    x = range(len(size_order))
    width = .78 / len(native)
    fig, ax = plt.subplots(figsize=(8.7, 5.1))
    offsets = np.arange(len(native)) - (len(native) - 1) / 2
    for offset, key, label, color in zip(offsets, native, labels, colors):
        table = size_selected[size_selected.model_key == key].set_index("size_band")
        values = [table.loc[band, "recall"] for band in size_order]
        bars = ax.bar([i + offset * width for i in x], values, width, label=label,
                      color=color, edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=10)
    ax.set(
        xticks=list(x), xticklabels=["Pequeño", "Mediano", "Grande"],
        ylabel="Recall de cajas de fuego", ylim=(0, 1),
        title="Recall de fuego por tamaño al punto operativo",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#E1E4E6"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figures / f"04_fire_recall_by_size.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)


def write_summary(output, selected, image_detection, paired, resources, standard, config):
    native_keys = []
    native_labels = {}
    for variant, spec in sorted(
        config["training_variants"].items(), key=lambda item: int(item[1]["trained_imgsz"])
    ):
        size = int(spec["trained_imgsz"])
        key = f"{variant}_eval{size}"
        native_keys.append(key)
        native_labels[key] = f"{size} → {size}"

    native = selected.set_index("model_key")
    images = image_detection.set_index("configuration")
    resources_by_variant = resources.set_index("variant")
    all_speed = standard[standard.scope == "all"].set_index("configuration")
    fire_standard = standard[standard.scope == "fire"].set_index("configuration")
    baseline_key = native_keys[0]
    baseline = native.loc[baseline_key]
    baseline_images = images.loc[baseline_key]
    baseline_speed = all_speed.loc[baseline_key]

    lines = [
        "# Comparación de resolución: YOLOv8s a 640, 768 y 1024", "",
        "## Resultado operativo", "",
        "Comparación sobre las mismas 1.721 imágenes de validación. Cada configuración elige su umbral "
        "maximizando el recall medio de humo y fuego con un máximo del 2 % de imágenes negativas con alarma.", "",
        "| Configuración nativa | Umbral | Recall humo | Recall fuego | Recall imágenes con fuego | Precisión micro | F1 micro | Negativas con alarma |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in native_keys:
        row, image_row = native.loc[key], images.loc[key]
        lines.append(
            f"| {native_labels[key]} | {row.threshold:.2f} | {row.smoke_recall:.2%} | {row.fire_recall:.2%} | "
            f"{image_row.fire_image_recall:.2%} | {row.micro_precision:.2%} | {row.micro_f1:.2%} | "
            f"{int(row.negative_images_with_alarm)}/{int(row.negative_images)} ({row.negative_alarm_rate:.2%}) |"
        )

    lines += ["", "## Cambios frente a 640 → 640", ""]
    for key in native_keys[1:]:
        row, image_row = native.loc[key], images.loc[key]
        variant = key.rsplit("_eval", 1)[0]
        inference_ratio = all_speed.loc[key, "inference_ms"] / baseline_speed.inference_ms
        lines += [
            f"### {native_labels[key]}", "",
            f"- Recall de cajas de fuego: {(row.fire_recall - baseline.fire_recall):+.2%}.",
            f"- Recall de imágenes con fuego: {(image_row.fire_image_recall - baseline_images.fire_image_recall):+.2%}.",
            f"- Precisión micro: {(row.micro_precision - baseline.micro_precision):+.2%}.",
            f"- F1 micro: {(row.micro_f1 - baseline.micro_f1):+.2%}.",
            f"- mAP50-95 estándar de fuego: {(fire_standard.loc[key, 'mAP50_95'] - fire_standard.loc[baseline_key, 'mAP50_95']):+.2%}.",
            f"- Tiempo de entrenamiento: {resources_by_variant.loc[variant, 'training_time_ratio_vs_640']:.2f}×.",
            f"- Inferencia de validación por imagen: {inference_ratio:.2f}×; es una medida por lotes.", "",
        ]

    focus_key = f"{config['focus_variant']}_eval{int(config['focus_evaluation_size'])}"
    reported_pairs = set(native_keys[1:] + [focus_key])
    lines += ["## Comparación emparejada frente a 640 → 640", ""]
    for row in paired[paired.candidate_key.isin(reported_pairs)].itertuples():
        if row.candidate_key in native_labels:
            label = native_labels[row.candidate_key]
        else:
            variant, eval_size = row.candidate_key.rsplit("_eval", 1)
            label = f"{variant.removeprefix('train')} → {eval_size}"
        unit = "imágenes con fuego" if row.unit == "fire_positive_image" else "cajas de fuego"
        lines.append(
            f"- {label}, {unit}: {row.gains} recuperaciones, {row.losses} pérdidas, "
            f"p exacta={row.p_value:.4g}, n={row.n}."
        )
    lines += [
        "", "## Alcance", "",
        "La cuadrícula 3×3 separa el efecto de la resolución usada durante el entrenamiento del efecto de la resolución de inferencia. "
        "Las pruebas emparejadas comparan las mismas imágenes o cajas, pero no incorporan la variabilidad entre semillas de entrenamiento.", "",
        "La rentabilidad se juzga por la detección bajo el mismo presupuesto de falsas alarmas frente al coste medido. "
        "No se ha consultado el test ni se ha cambiado el dataset.", "",
        "## Artefactos", "",
        "- `threshold_metrics.csv`: barrido completo de las nueve combinaciones.",
        "- `scenario_candidates.csv`: candidatos bajo el límite del 2 %.",
        "- `standard_metrics.csv`: métricas estándar de Ultralytics.",
        "- `image_detection_metrics.csv`: detección por imagen positiva.",
        "- `paired_comparison.csv`: cambios emparejados de todas las configuraciones frente al baseline.",
        "- `size_metrics.csv`: recall de cajas por tamaño.",
        "- `error_review_summary.csv`: desglose de falsos positivos y falsos negativos.",
        "- `error_review_examples.csv`: 25 imágenes prioritarias para revisión del modelo a 768.",
        "- `figures/`: gráficos PNG/SVG y ejemplos cualitativos 640 frente a 768.", "",
    ]
    (Path(output) / "RESUMEN_RESOLUCION.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path=None):
    import torch

    config_path = Path(config_path or ROOT / "configs" / "resolution_comparison.yaml")
    config = load_config(config_path)
    contract = pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])
    staged = pipeline.stage_prepared_dataset(contract, workers=int(config["staging_workers"]))
    manifest = pipeline.rebased_manifest(contract, staged)
    val_manifest = manifest[manifest.split == "val"].copy().reset_index(drop=True)
    records = {r["filename"]: r for r in val_manifest.to_dict("records")}

    experiments = {}
    for variant, spec in config["training_variants"].items():
        experiment = pipeline.resolve_experiment(experiment_id=spec["experiment_id"], root=ROOT)
        if experiment["status"] != "complete" or experiment["dataset_version"] != config["dataset_version"]:
            raise ValueError(f"Experimento no utilizable: {spec['experiment_id']}")
        saved_train_config = experiment.get("train_config")
        saved_imgsz = (
            saved_train_config.get("imgsz", spec["trained_imgsz"])
            if isinstance(saved_train_config, dict) else spec["trained_imgsz"]
        )
        if int(saved_imgsz) != int(spec["trained_imgsz"]):
            raise ValueError(f"Resolución de entrenamiento incoherente: {spec['experiment_id']}")
        experiments[variant] = experiment

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + sha256_json(config)[:8]
    parent = ROOT / "artifacts" / "06_resolution_comparison" / "validation"
    output = parent / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.yaml")
    code_dir = output / "code"; code_dir.mkdir()
    for path in (Path(__file__), ROOT / "tfm_thresholds.py", ROOT / "tfm_evaluation.py", ROOT / "tfm_pipeline.py"):
        shutil.copy2(path, code_dir / path.name)
    summary = {
        "schema_version": 1, "status": "running", "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"], "split": "val",
        "manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
        "config": config, "test_inference_executed": False,
    }
    pipeline.write_json_atomic(output / "run_summary.json", summary)

    all_standard, all_metrics, all_images, all_sizes, all_gt = [], [], [], [], []
    payload_map, selected_images, selected_gt = {}, {}, {}
    try:
        for variant, experiment in experiments.items():
            trained_imgsz = int(config["training_variants"][variant]["trained_imgsz"])
            for eval_imgsz in config["evaluation_sizes"]:
                key = f"{variant}_eval{eval_imgsz}"
                print(f"\n=== {key} ===", flush=True)
                standard = standard_validation(experiment, contract, staged, config, eval_imgsz)
                for metric in standard["metrics"]:
                    all_standard.append({
                        "configuration": key, "variant": variant,
                        "trained_imgsz": trained_imgsz, "eval_imgsz": int(eval_imgsz),
                        **metric,
                        "preprocess_ms": standard["speed_ms_per_image"].get("preprocess"),
                        "inference_ms": standard["speed_ms_per_image"].get("inference"),
                        "postprocess_ms": standard["speed_ms_per_image"].get("postprocess"),
                        "peak_cuda_allocated_gib": standard["peak_cuda_allocated_gib"],
                    })
                combo_config = threshold_config(config, eval_imgsz)
                cache = get_prediction_cache(experiment, contract, combo_config)
                payloads = load_predictions(ROOT / cache["predictions_rel"], val_manifest, config["prediction_confidence"])
                payload_map[key] = payloads
                metrics, images, sizes = sweep_model(payloads, val_manifest, combo_config, key)
                all_metrics.append(metrics); all_images.append(images); all_sizes.append(sizes)

        metrics = pd.concat(all_metrics, ignore_index=True)
        images = pd.concat(all_images, ignore_index=True)
        sizes = pd.concat(all_sizes, ignore_index=True)
        standard = pd.DataFrame(all_standard)
        scenarios = choose_scenarios(metrics, config)
        selected = scenarios[(scenarios.scenario == "alarm_02pct") & scenarios.feasible].copy()
        expected_configurations = len(experiments) * len(config["evaluation_sizes"])
        if len(selected) != expected_configurations:
            raise AssertionError("No hay un candidato factible para cada combinación.")

        detection_rows = []
        for row in selected.itertuples():
            key = row.model_key
            selected_table = images[(images.model_key == key) & np.isclose(images.threshold, row.threshold)].copy()
            variant, eval_text = key.rsplit("_eval", 1)
            detection_rows.append({
                "configuration": key, "variant": variant,
                "trained_imgsz": int(config["training_variants"][variant]["trained_imgsz"]),
                "eval_imgsz": int(eval_text), "threshold": float(row.threshold),
                **image_detection_metrics(selected_table),
            })
            detail_config = threshold_config(config, int(eval_text))
            _, _, _, gt, _ = evaluate_threshold(payload_map[key], records, float(row.threshold), detail_config, key, details=True)
            selected_images[key] = selected_table
            selected_gt[key] = gt
            all_gt.append(gt)
        image_detection = pd.DataFrame(detection_rows)
        baseline_variant = config["comparison_baseline"]
        baseline_size = int(config["training_variants"][baseline_variant]["trained_imgsz"])
        baseline_key = f"{baseline_variant}_eval{baseline_size}"
        paired_tables, paired_fire_tables = [], []
        for candidate_key in sorted(selected_images):
            if candidate_key == baseline_key:
                continue
            comparison, fire_boxes = paired_native_comparison(
                selected_images, selected_gt, baseline_key, candidate_key
            )
            paired_tables.append(comparison)
            paired_fire_tables.append(fire_boxes)
        paired = pd.concat(paired_tables, ignore_index=True)
        paired_fire = pd.concat(paired_fire_tables, ignore_index=True)
        resources = training_resources(experiments, config)
        error_summary, error_examples = error_review_tables(selected_images, config)

        metrics.to_csv(output / "threshold_metrics.csv", index=False)
        images.to_csv(output / "image_metrics.csv", index=False)
        sizes.to_csv(output / "size_metrics.csv", index=False)
        scenarios.to_csv(output / "scenario_candidates.csv", index=False)
        standard.to_csv(output / "standard_metrics.csv", index=False)
        image_detection.to_csv(output / "image_detection_metrics.csv", index=False)
        pd.concat(all_gt, ignore_index=True).to_csv(output / "selected_ground_truth.csv", index=False)
        paired.to_csv(output / "paired_comparison.csv", index=False)
        paired_fire.to_csv(output / "paired_fire_boxes.csv", index=False)
        resources.to_csv(output / "training_resources.csv", index=False)
        error_summary.to_csv(output / "error_review_summary.csv", index=False)
        error_examples.to_csv(output / "error_review_examples.csv", index=False)

        build_figures(metrics, scenarios, standard, sizes, image_detection, config, output)
        thresholds = selected.set_index("model_key").threshold.to_dict()
        focus_variant = config["focus_variant"]
        focus_eval_size = int(config["focus_evaluation_size"])
        focus_key = f"{focus_variant}_eval{focus_eval_size}"
        focus_paired_fire = paired_fire[paired_fire.candidate_key == focus_key].copy()
        review = draw_gallery(
            payload_map, thresholds, focus_paired_fire, records, output,
            baseline_key, focus_key, int(config["gallery_pairs"]),
        )
        review.to_csv(output / "review_selection.csv", index=False)
        write_summary(output, selected, image_detection, paired, resources, standard, config)

        summary.update({
            "status": "complete", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "images_per_configuration": len(val_manifest), "negative_images": int((val_manifest.box_count == 0).sum()),
            "configurations": expected_configurations, "threshold_points": len(metrics),
            "environment": pipeline.environment_snapshot(),
            "inputs": {variant: {
                "experiment_id": exp["experiment_id"],
                "checkpoint_sha256": pipeline.sha256_file(Path(exp["best_model"])),
            } for variant, exp in experiments.items()},
            "output_hashes": {p.relative_to(output).as_posix(): pipeline.sha256_file(p)
                              for p in sorted(output.rglob("*")) if p.is_file() and p.name != "run_summary.json"},
        })
        pipeline.write_json_atomic(output / "run_summary.json", summary)
        pipeline.write_json_atomic(parent / "latest.json", {
            "run_id": run_id, "run_rel": pipeline.project_relative(output, ROOT),
            "summary_sha256": pipeline.sha256_file(output / "run_summary.json"),
        })
        print(f"\nComparación completa: {output}", flush=True)
        print(selected[["model_key", "threshold", "smoke_recall", "fire_recall", "micro_precision", "micro_f1", "negative_alarm_rate"]].to_string(index=False))
        return output
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        pipeline.write_json_atomic(output / "run_summary.json", summary)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "resolution_comparison.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
