"""Cierra la selección YOLO26s frente a YOLOv8s usando solo validación."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import tfm_pipeline as pipeline
from tfm_thresholds import evaluate_threshold, load_predictions
from tools.run_class_threshold_and_fp_review import build_grid


ROOT = pipeline.project_root()


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_latest(relative_path: str) -> tuple[Path, dict]:
    pointer_path = ROOT / relative_path
    pointer = pipeline.read_json(pointer_path)
    run_path = ROOT / pointer["run_rel"]
    if not run_path.is_dir():
        raise FileNotFoundError(f"No existe la ejecución indicada por {pointer_path}: {run_path}")
    return run_path, pointer


def validate_source(run_path: Path, dataset_version: str) -> dict:
    summary = pipeline.read_json(run_path / "run_summary.json")
    if summary.get("status") != "complete":
        raise ValueError(f"Artefacto incompleto: {run_path}")
    if summary.get("split") != "val" or summary.get("test_inference_executed") is not False:
        raise ValueError(f"La fuente no está limitada a validación: {run_path}")
    if summary.get("dataset_version") != dataset_version:
        raise ValueError(f"Dataset incompatible en {run_path}")
    verification = pipeline.read_json(run_path / "verification_report.json")
    if verification.get("status") != "passed" or verification.get("test_inference_executed") is not False:
        raise ValueError(f"Verificación de fuente no superada: {run_path}")
    return summary


def markdown_table(frame: pd.DataFrame, percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                values.append("—")
            elif column in percent_columns:
                values.append(f"{100 * float(value):.2f}%")
            elif column == "p exacta" and float(value) < 0.0001:
                values.append(f"{float(value):.2e}")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def select_inputs(config: dict, architecture_run: Path, threshold_run: Path, threshold_summary: dict):
    standard_source = pd.read_csv(architecture_run / "standard_metrics.csv")
    paired_source = pd.read_csv(architecture_run / "paired_comparison.csv")
    training_source = pd.read_csv(architecture_run / "training_resources.csv")
    geometric_source = pd.read_csv(threshold_run / "fp_geometric_summary.csv")
    manual_source = pd.read_csv(threshold_run / "fp_manual_review.csv")
    contract = pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])
    manifest = pipeline.rebased_manifest(contract)
    manifest = manifest[manifest.split == "val"].copy().reset_index(drop=True)
    records = {row["filename"]: row for row in manifest.to_dict("records")}
    search_config = {
        **threshold_summary["config"],
        **config["threshold_search"],
        "alarm_budget": float(config["selection_policy"]["maximum_negative_alarm_rate"]),
        "size_area_boundaries": config["size_area_boundaries"],
    }

    standard_rows, operating_rows, class_rows, size_rows, error_rows, grid_rows = [], [], [], [], [], []
    for candidate_key, candidate in config["candidates"].items():
        label = candidate["label"]
        configuration = candidate["configuration"]
        selected_standard = standard_source[standard_source.configuration == configuration].copy()
        if set(selected_standard.scope) != {"all", "smoke", "fire"}:
            raise ValueError(f"Métricas estándar incompletas para {configuration}")
        selected_standard.insert(0, "candidate", candidate_key)
        selected_standard.insert(1, "label", label)
        standard_rows.append(selected_standard)

        profile_key = candidate["threshold_profile"]
        cache = threshold_summary["inputs"][profile_key]
        payloads = load_predictions(
            ROOT / cache["predictions_rel"], manifest, search_config["prediction_confidence"]
        )
        eval_config = {
            **search_config,
            "imgsz": threshold_summary["config"]["profiles"][profile_key]["imgsz"],
        }
        grid, _ = build_grid(profile_key, payloads, manifest, search_config, eval_config)
        grid.insert(0, "candidate", candidate_key)
        grid.insert(1, "label", label)
        grid_rows.append(grid)
        feasible = grid[
            grid.negative_alarm_rate
            <= float(config["selection_policy"]["maximum_negative_alarm_rate"]) + 1e-12
        ]
        if feasible.empty:
            raise ValueError(f"No hay umbrales que cumplan el 1 % para {candidate_key}")
        profile = feasible.sort_values(
            ["macro_recall", "minimum_class_recall", "micro_f1", "micro_precision"],
            ascending=False,
            kind="stable",
        ).iloc[0]
        for field in ("smoke_threshold", "fire_threshold"):
            if not np.isclose(float(profile[field]), float(candidate[field])):
                raise ValueError(f"{field} no coincide para {candidate_key}")
        operating_rows.append({**profile.to_dict(), "candidate": candidate_key, "label": label,
                               "images": len(manifest),
                               "scenario": "max_macro_recall_01pct"})

        for class_name in ("smoke", "fire"):
            class_rows.append({
                "candidate": candidate_key,
                "label": label,
                "class_name": class_name,
                "threshold": float(profile[f"{class_name}_threshold"]),
                "tp": int(profile[f"{class_name}_tp"]),
                "fp": int(profile[f"{class_name}_fp"]),
                "fn": int(profile[f"{class_name}_fn"]),
                "precision": float(profile[f"{class_name}_precision"]),
                "recall": float(profile[f"{class_name}_recall"]),
                "f1": float(profile[f"{class_name}_f1"]),
            })
            threshold = float(profile[f"{class_name}_threshold"])
            _, _, all_sizes, _, _ = evaluate_threshold(
                payloads, records, threshold, eval_config, configuration
            )
            selected_size = all_sizes[all_sizes.class_name == class_name].copy()
            if set(selected_size.size_band) != {"small", "medium", "large"}:
                raise ValueError(f"Desglose por tamaño incompleto: {candidate_key}/{class_name}/{threshold}")
            selected_size.insert(0, "candidate", candidate_key)
            selected_size.insert(1, "label", label)
            size_rows.append(selected_size)

        error_rows.append({
            "candidate": candidate_key,
            "label": label,
            "smoke_fp": int(profile.smoke_fp), "smoke_fn": int(profile.smoke_fn),
            "fire_fp": int(profile.fire_fp), "fire_fn": int(profile.fire_fn),
            "positive_image_fp_boxes": int(profile.positive_image_fp_boxes),
            "negative_image_fp_boxes": int(profile.negative_image_fp_boxes),
            "negative_images_with_alarm": int(profile.negative_images_with_alarm),
        })

    standard = pd.concat(standard_rows, ignore_index=True)
    operating = pd.DataFrame(operating_rows)
    class_metrics = pd.DataFrame(class_rows)
    size_metrics = pd.concat(size_rows, ignore_index=True)
    errors = pd.DataFrame(error_rows)
    paired = paired_source[paired_source.eval_imgsz == 768].copy()
    training = training_source.copy()
    manual_summary = (
        manual_source.groupby("manual_category", as_index=False)
        .agg(images=("review_id", "count"), observation_example=("observation", "first"))
        .sort_values(["images", "manual_category"], ascending=[False, True])
    )
    threshold_grid = pd.concat(grid_rows, ignore_index=True)
    return (standard, operating, class_metrics, size_metrics, errors, paired, training,
            geometric_source, manual_summary, threshold_grid)


def rank_candidates(operating: pd.DataFrame, config: dict) -> pd.DataFrame:
    policy = config["selection_policy"]
    ranked = operating.copy()
    ranked["feasible_final"] = ranked.negative_alarm_rate <= float(policy["maximum_negative_alarm_rate"])
    sort_columns = ["feasible_final", policy["primary_metric"], *policy["tie_breakers"]]
    ranked = ranked.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    ranked["selected"] = ranked.selection_rank == 1
    if not bool(ranked.iloc[0].feasible_final):
        raise ValueError("Ningún candidato cumple el presupuesto de falsas alarmas")
    return ranked


def compare_alarm_budgets(threshold_grid: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for candidate_key, candidate in config["candidates"].items():
        candidate_grid = threshold_grid[threshold_grid.candidate == candidate_key]
        for budget in config["comparison_alarm_budgets"]:
            feasible = candidate_grid[candidate_grid.negative_alarm_rate <= float(budget) + 1e-12]
            if feasible.empty:
                raise ValueError(f"Sin punto factible para {candidate_key} al {budget:.1%}")
            selected = feasible.sort_values(
                ["macro_recall", "minimum_class_recall", "micro_f1", "micro_precision"],
                ascending=False, kind="stable",
            ).iloc[0]
            rows.append({
                "candidate": candidate_key, "label": candidate["label"],
                "alarm_budget": float(budget), **selected.to_dict(),
            })
    return pd.DataFrame(rows)


def add_bar_labels(axis, values, percent=True):
    for patch, value in zip(axis.patches, values):
        text = f"{100 * value:.1f}%" if percent else f"{value:.0f}"
        axis.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.012, text,
                  ha="center", va="bottom", fontsize=9)


def build_figures(output: Path, operating: pd.DataFrame, class_metrics: pd.DataFrame,
                  size_metrics: pd.DataFrame, errors: pd.DataFrame,
                  budget_comparison: pd.DataFrame, config: dict) -> list[Path]:
    figure_dir = output / "figures"
    figure_dir.mkdir()
    colors = {key: value["color"] for key, value in config["candidates"].items()}
    labels = {key: value["label"] for key, value in config["candidates"].items()}
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    paths = []

    metrics = ["micro_precision", "micro_recall", "micro_f1", "macro_recall"]
    metric_labels = ["Precisión micro", "Recall micro", "F1 micro", "Recall macro"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(metrics)); width = 0.34
    label_values = []
    for index, row in operating.reset_index(drop=True).iterrows():
        values = [float(row[m]) for m in metrics]
        label_values.extend(values)
        axes[0].bar(x + (index - 0.5) * width, values, width, label=labels[row.candidate],
                    color=colors[row.candidate])
        for xpos, value in zip(x + (index - 0.5) * width, values):
            axes[0].text(xpos, value + 0.015, f"{100*value:.1f}%", ha="center", fontsize=9)
    axes[0].set_ylim(0, 1.0); axes[0].set_xticks(x, metric_labels)
    axes[0].set_ylabel("Proporción"); axes[0].set_title("Métricas en los puntos operativos elegidos")
    axes[0].grid(axis="y", color="#D9DEE3", linewidth=0.8); axes[0].legend(loc="lower right")
    alarm = operating.set_index("candidate").loc[list(config["candidates"]), "negative_alarm_rate"]
    bars = axes[1].bar([labels[key] for key in alarm.index], alarm.values,
                       color=[colors[key] for key in alarm.index])
    alarm_budget = float(config["selection_policy"]["maximum_negative_alarm_rate"])
    axes[1].axhline(alarm_budget, color="#333333", linestyle="--",
                    label=f"Límite {100*alarm_budget:.0f} %")
    axes[1].set_ylim(0, alarm_budget * 1.25); axes[1].set_ylabel("Imágenes negativas con alarma")
    axes[1].set_title("Presupuesto de falsas alarmas")
    axes[1].grid(axis="y", color="#D9DEE3", linewidth=0.8); axes[1].legend()
    for bar, value in zip(bars, alarm.values):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+0.0007, f"{100*value:.2f}%", ha="center")
    fig.suptitle("Comparación final en validación D-Fire", fontsize=17)
    fig.tight_layout(); path = figure_dir / "01_global_operating_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    candidates = list(config["candidates"]); width = 0.34
    classes = ["smoke", "fire"]
    for index, key in enumerate(candidates):
        subset = class_metrics[class_metrics.candidate == key].set_index("class_name").loc[classes]
        positions = np.arange(2) + (index - 0.5) * width
        axes[0].bar(positions, subset.recall, width, color=colors[key], label=labels[key])
        for xpos, value in zip(positions, subset.recall):
            axes[0].text(xpos, value+0.015, f"{100*value:.1f}%", ha="center", fontsize=9)
    axes[0].set_xticks(np.arange(2), ["Humo", "Fuego"]); axes[0].set_title("Recall por clase")
    size_order = ["small", "medium", "large"]
    size_labels = ["Pequeño", "Mediano", "Grande"]
    for axis, class_name, title in zip(axes[1:], classes, ["Humo por tamaño", "Fuego por tamaño"]):
        for index, key in enumerate(candidates):
            subset = size_metrics[(size_metrics.candidate == key) & (size_metrics.class_name == class_name)]
            subset = subset.set_index("size_band").loc[size_order]
            positions = np.arange(3) + (index - 0.5) * width
            axis.bar(positions, subset.recall, width, color=colors[key], label=labels[key])
            for xpos, value in zip(positions, subset.recall):
                axis.text(xpos, value+0.015, f"{100*value:.1f}%", ha="center", fontsize=8)
        axis.set_xticks(np.arange(3), size_labels); axis.set_title(title)
    for axis in axes:
        axis.set_ylim(0, 1.0); axis.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    axes[0].set_ylabel("Recall de cajas"); axes[2].legend(loc="lower right")
    fig.suptitle("Cobertura por clase y tamaño en validación", fontsize=17)
    fig.tight_layout(); path = figure_dir / "02_class_and_size_recall.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    x = np.arange(2)
    for index, key in enumerate(candidates):
        row = errors[errors.candidate == key].iloc[0]
        fp = [row.smoke_fp, row.fire_fp]; fn = [row.smoke_fn, row.fire_fn]
        positions = x + (index - 0.5) * width
        axes[0].bar(positions, fp, width, color=colors[key], label=labels[key])
        axes[1].bar(positions, fn, width, color=colors[key], label=labels[key])
        for axis, positions_here, values in ((axes[0], positions, fp), (axes[1], positions, fn)):
            for xpos, value in zip(positions_here, values):
                axis.text(xpos, value + 12, str(int(value)), ha="center", fontsize=9)
    for axis, title in zip(axes, ["Falsos positivos", "Falsos negativos"]):
        axis.set_xticks(x, ["Humo", "Fuego"]); axis.set_title(title)
        axis.set_ylabel("Cajas"); axis.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    axes[1].legend(loc="upper left")
    fig.suptitle("Perfil de errores en los puntos operativos", fontsize=17)
    fig.tight_layout(); path = figure_dir / "03_fp_fn_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    budgets = list(config["comparison_alarm_budgets"])
    x = np.arange(len(budgets)); width = 0.34
    for index, key in enumerate(candidates):
        subset = budget_comparison[budget_comparison.candidate == key].set_index("alarm_budget").loc[budgets]
        positions = x + (index - 0.5) * width
        axes[0].bar(positions, subset.macro_recall, width, color=colors[key], label=labels[key])
        axes[1].bar(positions, subset.micro_precision, width, color=colors[key], label=labels[key])
        for axis, positions_here, values in (
            (axes[0], positions, subset.macro_recall),
            (axes[1], positions, subset.micro_precision),
        ):
            for xpos, value in zip(positions_here, values):
                axis.text(xpos, value + 0.015, f"{100*value:.1f}%", ha="center", fontsize=9)
    for axis, title in zip(axes, ["Recall macro", "Precisión micro"]):
        axis.set_xticks(x, [f"Límite {100*b:.0f} %" for b in budgets])
        axis.set_ylim(0, 1.0); axis.set_ylabel("Proporción"); axis.set_title(title)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    axes[1].legend(loc="lower right")
    fig.suptitle("Coste de endurecer el límite de alarmas negativas", fontsize=17)
    fig.tight_layout(); path = figure_dir / "04_alarm_budget_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def write_summary(output: Path, ranked: pd.DataFrame, standard: pd.DataFrame,
                  class_metrics: pd.DataFrame, size_metrics: pd.DataFrame,
                  errors: pd.DataFrame, paired: pd.DataFrame, training: pd.DataFrame,
                  manual_summary: pd.DataFrame, budget_comparison: pd.DataFrame,
                  config: dict) -> Path:
    winner = ranked.iloc[0]
    global_table = ranked[["selection_rank", "label", "smoke_threshold", "fire_threshold",
                           "micro_precision", "micro_recall", "micro_f1", "macro_recall",
                           "minimum_class_recall", "negative_images_with_alarm",
                           "negative_images", "negative_alarm_rate"]].copy()
    global_table.columns = ["Rango", "Modelo", "Umbral humo", "Umbral fuego", "Precisión micro",
                            "Recall micro", "F1 micro", "Recall macro", "Recall mínimo",
                            "Negativas con alarma", "Negativas", "Tasa alarma negativa"]
    class_table = class_metrics[["label", "class_name", "threshold", "precision", "recall", "f1", "tp", "fp", "fn"]].copy()
    class_table.columns = ["Modelo", "Clase", "Umbral", "Precisión", "Recall", "F1", "TP", "FP", "FN"]
    class_table["Clase"] = class_table["Clase"].map({"smoke": "Humo", "fire": "Fuego"})
    size_table = size_metrics[["label", "class_name", "size_band", "threshold", "gt_boxes", "tp", "fn", "recall"]].copy()
    size_table.columns = ["Modelo", "Clase", "Tamaño", "Umbral", "Cajas GT", "TP", "FN", "Recall"]
    size_table["Clase"] = size_table["Clase"].map({"smoke": "Humo", "fire": "Fuego"})
    size_table["Tamaño"] = size_table["Tamaño"].map({"small": "Pequeño", "medium": "Mediano", "large": "Grande"})
    paired_table = paired[["class_name", "unit", "n", "baseline_rate", "candidate_rate", "gains", "losses", "p_value"]].copy()
    paired_table.columns = ["Clase", "Unidad", "n", "YOLOv8s", "YOLO26s", "Ganancias", "Pérdidas", "p exacta"]
    standard_all = standard[standard.scope == "all"][["label", "precision", "recall", "mAP50", "mAP50_95", "inference_ms"]].copy()
    standard_all.columns = ["Modelo", "Precisión", "Recall", "mAP50", "mAP50-95", "Inferencia ms"]
    error_table = errors[["label", "smoke_fp", "smoke_fn", "fire_fp", "fire_fn",
                          "positive_image_fp_boxes", "negative_image_fp_boxes"]].copy()
    error_table.columns = ["Modelo", "FP humo", "FN humo", "FP fuego", "FN fuego",
                           "FP en positivas", "FP en negativas"]
    budget_table = budget_comparison[[
        "label", "alarm_budget", "smoke_threshold", "fire_threshold",
        "micro_precision", "micro_recall", "micro_f1", "macro_recall",
        "negative_images_with_alarm", "negative_alarm_rate",
    ]].copy()
    budget_table.columns = ["Modelo", "Presupuesto", "Umbral humo", "Umbral fuego",
                            "Precisión micro", "Recall micro", "F1 micro", "Recall macro",
                            "Negativas con alarma", "Tasa alarma negativa"]
    percent = {"Precisión micro", "Recall micro", "F1 micro", "Recall macro", "Recall mínimo",
               "Tasa alarma negativa", "Precisión", "Recall", "F1", "YOLOv8s", "YOLO26s",
               "mAP50", "mAP50-95", "Presupuesto"}
    text = f"""# Selección final de modelo sobre validación

## tl;dr

Se selecciona **{winner['label']}** con umbrales **humo {winner.smoke_threshold:.2f}** y
**fuego {winner.fire_threshold:.2f}**. Cumple el límite de alarma negativa del 1 %
y obtiene el mayor recall macro ({100*winner.macro_recall:.2f} %) y el mayor recall
mínimo entre clases ({100*winner.minimum_class_recall:.2f} %). El conjunto de test
permanece cerrado y no se ha usado para esta decisión.

## Método y población

- 1.721 imágenes de validación congeladas; 783 completamente negativas.
- IoU de acierto 0,50 y predicciones base almacenadas a confianza 0,01.
- Cada modelo usa la resolución operativa elegida previamente: YOLO26s 768→768 y YOLOv8s 768→640.
- Primero se exige ≤1 % de imágenes negativas con alarma. Después se ordena por
  recall macro, recall de la clase más débil, F1 micro y precisión micro.
- Los umbrales por clase se eligen maximizando recall macro dentro de ese límite.
  Con 783 negativas, como máximo se admiten 7 imágenes con alarma: 8/783 ya
  equivaldría al 1,02 % y superaría el presupuesto.

## Tabla definitiva en el punto operativo

{markdown_table(global_table, percent)}

Ambos modelos quedan prácticamente empatados en precisión. YOLO26s se elige
porque obtiene mayor recall macro, mayor recall de la clase más débil y mayor F1
micro una vez satisfecho el presupuesto de falsas alarmas.

## Métricas estándar

{markdown_table(standard_all, percent)}

Las métricas estándar se muestran para contexto y no sustituyen la evaluación
en el punto operativo con umbrales por clase.

## Comparación entre presupuestos del 1 % y 2 %

{markdown_table(budget_table, percent)}

Endurecer el límite al 1 % reduce el recall macro aproximadamente tres puntos en
ambos modelos, sobre todo por el mayor umbral de humo. A cambio aumenta la
precisión micro y reduce de 12 a 7 alarmas negativas en YOLO26s y de 15 a 7 en
YOLOv8s. YOLO26s continúa siendo el modelo con mayor recall macro bajo ambos límites.

## Métricas por clase

{markdown_table(class_table, percent)}

## Recall por tamaño

Las bandas usan área normalizada: pequeño <1 %, mediano 1–10 % y grande ≥10 %.

{markdown_table(size_table, percent)}

YOLO26s presenta mayor recall en las tres bandas de humo y en fuego grande;
empata en fuego mediano y queda 0,40 puntos por debajo en fuego pequeño.

## Falsos positivos y falsos negativos

{markdown_table(error_table)}

YOLO26s reduce en 36 los falsos negativos de humo y empata en los de fuego. Los
dos candidatos activan 7 de las 783 imágenes negativas; la mayor parte de sus cajas FP aparece en
imágenes que sí contienen humo o fuego y debe interpretarse junto a la revisión
de localización, duplicados y anotaciones incompletas.

La inspección manual disponible revisó 27 ejemplos del perfil sensible YOLO26s
a umbral común 0,16; es diagnóstica y no representa exactamente el punto final
{winner.smoke_threshold:.2f}/{winner.fire_threshold:.2f}. Encontró principalmente diferencias de extensión/localización,
duplicados, posible anotación incompleta, ambigüedad de clase, luces artificiales
y objetos o reflejos rojos. No apareció una nube meteorológica inequívoca en esa muestra.

## Comparación emparejada a resolución común 768

{markdown_table(paired_table, percent)}

Esta comparación previa eligió puntos bajo el presupuesto común del 2 % y se
mantiene únicamente como evidencia complementaria a resolución controlada.
YOLO26s recupera más cajas e imágenes en las cuatro comparaciones a 768. Las
cuatro diferencias tienen p<0,05 sin corregir; con Bonferroni para cuatro pruebas
se mantienen tres y deja de superar el umbral la detección de imágenes con fuego.
La significación no se ha usado como criterio directo de selección.

## Decisión

Se congela como candidato previo a test **YOLO26s 768→768**, checkpoint
`{config['candidates']['yolo26s']['experiment_id']}`, con humo
{winner.smoke_threshold:.2f} y fuego {winner.fire_threshold:.2f}.
No se afirma todavía rendimiento final: la siguiente evaluación debe ejecutar
esta configuración una única vez sobre test, sin reajustar pesos ni umbrales.
"""
    target = output / "RESUMEN_SELECCION_VALIDACION.md"
    target.write_text(text, encoding="utf-8")
    return target


def run(config_path: str | Path) -> Path:
    config_path = (ROOT / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    config = read_yaml(config_path)
    if config.get("split") != "val" or config.get("test_locked") is not True:
        raise ValueError("La selección final debe estar bloqueada a validación")
    architecture_run, architecture_pointer = resolve_latest(config["architecture_comparison_latest"])
    threshold_run, threshold_pointer = resolve_latest(config["class_threshold_review_latest"])
    architecture_summary = validate_source(architecture_run, config["dataset_version"])
    threshold_summary = validate_source(threshold_run, config["dataset_version"])

    config_hash = pipeline.sha256_file(config_path)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + config_hash[:8]
    parent = ROOT / config["output_parent"]
    output = parent / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        tables = select_inputs(config, architecture_run, threshold_run, threshold_summary)
        (standard, operating, class_metrics, size_metrics, errors, paired, training,
         geometric, manual, threshold_grid) = tables
        ranked = rank_candidates(operating, config)
        budget_comparison = compare_alarm_budgets(threshold_grid, config)
        standard.to_csv(output / "standard_metrics.csv", index=False)
        ranked.to_csv(output / "final_operating_points.csv", index=False)
        class_metrics.to_csv(output / "class_metrics.csv", index=False)
        size_metrics.to_csv(output / "size_metrics.csv", index=False)
        errors.to_csv(output / "error_comparison.csv", index=False)
        paired.to_csv(output / "paired_comparison_768.csv", index=False)
        training.to_csv(output / "training_resources.csv", index=False)
        geometric.to_csv(output / "yolo26_fp_geometric_summary.csv", index=False)
        manual.to_csv(output / "manual_fp_review_summary.csv", index=False)
        threshold_grid.to_csv(output / "class_threshold_grid_01pct.csv", index=False)
        budget_comparison.to_csv(output / "alarm_budget_comparison.csv", index=False)
        figures = build_figures(output, operating, class_metrics, size_metrics, errors,
                                budget_comparison, config)
        summary_path = write_summary(output, ranked, standard, class_metrics, size_metrics,
                                     errors, paired, training, manual, budget_comparison, config)
        shutil.copy2(config_path, output / "config.yaml")
        code_dir = output / "code"; code_dir.mkdir()
        shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
        verifier = ROOT / "tools" / "verify_final_validation_selection.py"
        if verifier.exists(): shutil.copy2(verifier, code_dir / verifier.name)

        run_summary = {
            "schema_version": 1, "status": "complete", "run_id": run_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "split": "val", "dataset_version": config["dataset_version"],
            "test_inference_executed": False,
            "selected_candidate": ranked.iloc[0].candidate,
            "selected_label": ranked.iloc[0].label,
            "selected_experiment_id": config["candidates"][ranked.iloc[0].candidate]["experiment_id"],
            "selected_smoke_threshold": float(ranked.iloc[0].smoke_threshold),
            "selected_fire_threshold": float(ranked.iloc[0].fire_threshold),
            "source_runs": {
                "architecture": architecture_pointer,
                "threshold_review": threshold_pointer,
            },
            "source_test_flags": [architecture_summary["test_inference_executed"], threshold_summary["test_inference_executed"]],
            "images": int(ranked.iloc[0].images),
            "negative_images": int(ranked.iloc[0].negative_images),
            "output_hashes": {
                str(path.relative_to(output)).replace("\\", "/"): pipeline.sha256_file(path)
                for path in output.rglob("*") if path.is_file() and path.name != "run_summary.json"
            },
            "figures": [str(path.relative_to(ROOT)).replace("\\", "/") for path in figures],
        }
        pipeline.write_json_atomic(output / "run_summary.json", run_summary)
        latest = {
            "run_id": run_id,
            "run_rel": str(output.relative_to(ROOT)).replace("\\", "/"),
            "summary_sha256": pipeline.sha256_file(summary_path),
        }
        pipeline.write_json_atomic(parent / "latest.json", latest)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(f"Selección final de validación: {output}")
    print(ranked[["selection_rank", "label", "micro_precision", "micro_recall", "micro_f1",
                  "macro_recall", "negative_alarm_rate", "selected"]].to_string(index=False))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_validation_selection.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
