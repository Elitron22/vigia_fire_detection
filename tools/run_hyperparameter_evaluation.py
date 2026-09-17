"""Evalúa automáticamente el cribado multifidelidad de YOLO26s en validación."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline
from tfm_thresholds import load_predictions
from tools.run_class_threshold_and_fp_review import build_grid
from tools.run_threshold_sweep import get_prediction_cache

DEFAULT_CONFIG = ROOT / "configs" / "yolo26s_hyperparameter_search.yaml"
OUTPUT_PARENT = ROOT / "artifacts" / "11_yolo26s_hyperparameter_search" / "validation"
TRAIN_METRICS = {
    "map5095": "metrics/mAP50-95(B)",
    "map50": "metrics/mAP50(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}
COLORS = {
    "baseline": "#275D8C",
    "hp01_lr5e4": "#C58A1B",
    "hp02_weight_decay1e3": "#B86A22",
    "hp03_conservative_aug": "#727B38",
    "hp04_lr5e4_conservative_aug": "#B55273",
}


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 2 or config.get("model_key") != "yolo26s":
        raise ValueError("La evaluación requiere schema_version=2 y model_key=yolo26s.")
    if not config.get("selection", {}).get("test_locked"):
        raise ValueError("test_locked debe permanecer activado.")
    if config["selection"].get("split") != "val" or config["evaluation"].get("split") != "val":
        raise ValueError("La selección y evaluación solo pueden utilizar val.")
    active = config.get("active_trials", [])
    if len(active) != 4 or len(active) != len(set(active)) or not set(active) <= set(config["trials"]):
        raise ValueError("Se requieren cuatro ensayos activos, únicos y declarados.")
    if config["screening_parameters"].get("epochs") != config.get("screening_epochs"):
        raise ValueError("El horizonte del cribado no coincide con su configuración.")
    evaluation = config["evaluation"]
    if evaluation.get("candidate_count") != 2:
        raise ValueError("El protocolo requiere exactamente dos candidatos preliminares.")
    if not 0 < evaluation["prediction_confidence"] <= 1:
        raise ValueError("prediction_confidence no válido.")
    if not 0 < evaluation["alarm_budget"] < 1:
        raise ValueError("alarm_budget no válido.")
    for field in ("smoke_thresholds", "fire_thresholds"):
        values = np.asarray(evaluation[field], dtype=float)
        if (not len(values) or np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0)
                or values[0] < evaluation["prediction_confidence"] or values[-1] > 1):
            raise ValueError(f"Malla no válida: {field}.")
    return config


def _resolve_descriptor(experiment_id: str) -> dict | None:
    path = pipeline.experiments_root(ROOT) / experiment_id / "experiment.json"
    if not path.exists():
        return None
    descriptor = pipeline.resolve_descriptor_paths(pipeline.read_json(path), ROOT)
    # get_prediction_cache stores its verified predictions below the experiment.
    # Descriptors persist portable relative paths, so this runtime-only field has
    # to be restored explicitly (as list_experiments already does).
    descriptor["experiment_root"] = str(path.parent.resolve())
    return descriptor


def resolve_experiments(config: dict, *, require_complete: bool) -> dict[str, dict | None]:
    experiments: dict[str, dict | None] = {
        "baseline": _resolve_descriptor(config["baseline_experiment_id"])
    }
    for trial_id in config["active_trials"]:
        experiments[trial_id] = _resolve_descriptor(config["trials"][trial_id]["experiment_id"])

    manifest_sha = pipeline.sha256_file(
        pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])["manifest_path"]
    )
    problems = []
    for trial_id, experiment in experiments.items():
        if experiment is None:
            problems.append(f"{trial_id}: ausente")
            continue
        if experiment.get("status") != "complete":
            problems.append(f"{trial_id}: {experiment.get('status', 'sin estado')}")
            continue
        if experiment.get("model_key") != "yolo26s":
            problems.append(f"{trial_id}: modelo distinto")
        if experiment.get("dataset_version") != config["dataset_version"]:
            problems.append(f"{trial_id}: dataset distinto")
        if experiment.get("dataset_manifest_sha256") != manifest_sha:
            problems.append(f"{trial_id}: manifiesto distinto")
        if not Path(experiment["best_model"]).exists():
            problems.append(f"{trial_id}: falta best.pt")
        results = Path(experiment["training_run_dir"]) / "results.csv"
        if not results.exists():
            problems.append(f"{trial_id}: falta results.csv")
        if trial_id != "baseline":
            expected = config["trials"][trial_id]
            if experiment.get("search_id") != config["search_id"]:
                problems.append(f"{trial_id}: search_id distinto")
            if experiment.get("trial_id") != trial_id:
                problems.append(f"{trial_id}: trial_id distinto")
            if experiment.get("seed") != expected["seed"]:
                problems.append(f"{trial_id}: semilla distinta")
            if experiment.get("train_config") != (
                dict(config["screening_parameters"]) | dict(expected.get("overrides", {}))
            ):
                problems.append(f"{trial_id}: hiperparámetros distintos")
    if require_complete and problems:
        raise RuntimeError("El cribado aún no es evaluable: " + "; ".join(problems))
    return experiments


def training_summary(experiments: dict[str, dict], config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = int(config["selection"]["screening_horizon_epochs"])
    rows, curves = [], []
    for trial_id, experiment in experiments.items():
        if experiment is None or experiment.get("status") != "complete":
            continue
        path = Path(experiment["training_run_dir"]) / "results.csv"
        results = pd.read_csv(path)
        results.columns = [column.strip() for column in results.columns]
        missing = (set(TRAIN_METRICS.values()) | {"epoch"}) - set(results)
        if missing:
            raise ValueError(f"Columnas ausentes en {path}: {sorted(missing)}")
        comparable = results.iloc[:horizon].copy()
        if len(comparable) != horizon or int(comparable.epoch.iloc[-1]) != horizon:
            raise ValueError(f"{trial_id} no tiene exactamente {horizon} épocas comparables.")
        primary = TRAIN_METRICS["map5095"]
        best_index = comparable[primary].astype(float).idxmax()
        best = comparable.loc[best_index]
        x = comparable.epoch.tail(10).to_numpy(dtype=float)
        y = comparable[primary].tail(10).to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan
        rows.append({
            "trial_id": trial_id,
            "experiment_id": experiment["experiment_id"],
            "best_epoch_50": int(best["epoch"]),
            "best_map5095_50": float(best[TRAIN_METRICS["map5095"]]),
            "best_map50_50": float(best[TRAIN_METRICS["map50"]]),
            "precision_at_best_50": float(best[TRAIN_METRICS["precision"]]),
            "recall_at_best_50": float(best[TRAIN_METRICS["recall"]]),
            "last5_mean_map5095": float(comparable[primary].tail(5).mean()),
            "last10_slope_map5095_per_epoch": slope,
            "epochs_compared": horizon,
            "epochs_available": len(results),
        })
        curve = comparable[["epoch", *TRAIN_METRICS.values()]].copy()
        curve.insert(0, "trial_id", trial_id)
        curves.append(curve)
    table = pd.DataFrame(rows)
    if "baseline" not in set(table.trial_id) or len(table) != 1 + len(config["active_trials"]):
        raise AssertionError("La tabla de entrenamiento no contiene baseline y cuatro ensayos.")
    baseline_map = float(table.loc[table.trial_id == "baseline", "best_map5095_50"].iloc[0])
    table["delta_map5095_vs_baseline"] = table.best_map5095_50 - baseline_map
    return table, pd.concat(curves, ignore_index=True)


def select_preliminary(training: pd.DataFrame, candidate_count: int = 2) -> pd.DataFrame:
    required = {"trial_id", "best_map5095_50", "last5_mean_map5095", "recall_at_best_50"}
    if not required <= set(training):
        raise ValueError(f"Faltan columnas de selección: {sorted(required - set(training))}")
    ranked = training.sort_values(
        ["best_map5095_50", "last5_mean_map5095", "recall_at_best_50", "trial_id"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "preliminary_rank", np.arange(1, len(ranked) + 1))
    ranked["selected_for_operational_evaluation"] = ranked.preliminary_rank <= candidate_count
    return ranked


def select_operating_points(grid: pd.DataFrame, alarm_budget: float) -> pd.DataFrame:
    selections = []
    for trial_id, table in grid.groupby("profile", sort=False):
        feasible = table[
            table.negative_images_with_alarm <= alarm_budget * table.negative_images + 1e-12
        ]
        if feasible.empty:
            raise AssertionError(f"{trial_id} no tiene puntos dentro del presupuesto de alarmas.")
        sensitivity = feasible.sort_values(
            ["macro_recall", "minimum_class_recall", "micro_f1", "micro_precision",
             "negative_images_with_alarm", "smoke_threshold", "fire_threshold"],
            ascending=[False, False, False, False, True, False, False], kind="stable",
        ).iloc[0]
        balanced = feasible.sort_values(
            ["micro_f1", "macro_recall", "micro_precision", "minimum_class_recall",
             "negative_images_with_alarm", "smoke_threshold", "fire_threshold"],
            ascending=[False, False, False, False, True, False, False], kind="stable",
        ).iloc[0]
        for scenario, selected in (("sensitivity", sensitivity), ("balanced", balanced)):
            selections.append({**selected.to_dict(), "trial_id": trial_id,
                               "scenario": scenario, "alarm_budget": alarm_budget})
    return pd.DataFrame(selections)


def choose_overall_winners(recommendations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, table in recommendations.groupby("scenario", sort=False):
        if scenario == "sensitivity":
            ordering = ["macro_recall", "minimum_class_recall", "micro_f1",
                        "negative_images_with_alarm", "trial_id"]
            ascending = [False, False, False, True, True]
        else:
            ordering = ["micro_f1", "macro_recall", "micro_precision",
                        "negative_images_with_alarm", "trial_id"]
            ascending = [False, False, False, True, True]
        winner = table.sort_values(ordering, ascending=ascending, kind="stable").iloc[0]
        rows.append(winner.to_dict())
    return pd.DataFrame(rows)


def _display_label(trial_id: str) -> str:
    return {
        "baseline": "Baseline",
        "hp01_lr5e4": "lr0 0,0005",
        "hp02_weight_decay1e3": "Weight decay 0,001",
        "hp03_conservative_aug": "Aumentos conservadores",
        "hp04_lr5e4_conservative_aug": "lr0 + aumentos",
    }.get(trial_id, trial_id)


def make_figures(training: pd.DataFrame, curves: pd.DataFrame, recommendations: pd.DataFrame,
                 grid: pd.DataFrame, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figures = output / "figures"
    figures.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10})

    ordered = training.sort_values("best_map5095_50")
    labels = [_display_label(value) for value in ordered.trial_id]
    colors = [COLORS.get(value, "#777777") for value in ordered.trial_id]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, ordered.best_map5095_50, color=colors)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in ordered.best_map5095_50], padding=3)
    ax.set(xlim=(0, 0.55), xlabel="Mejor mAP50-95 en validación",
           title="Cribado YOLO26s al mismo horizonte de 50 épocas")
    ax.grid(axis="x", color="#D9D9D9", alpha=.7)
    fig.tight_layout()
    path = figures / "01_screening_map5095.png"
    fig.savefig(path, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for trial_id, table in curves.groupby("trial_id", sort=False):
        ax.plot(table.epoch, table[TRAIN_METRICS["map5095"]], label=_display_label(trial_id),
                color=COLORS.get(trial_id, "#777777"), linewidth=2)
    ax.set(xlabel="Época", ylabel="mAP50-95 de validación", xlim=(1, 50), ylim=(0, .55),
           title="Convergencia durante el cribado multifidelidad")
    ax.grid(color="#D9D9D9", alpha=.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path2 = figures / "02_screening_curves.png"
    fig.savefig(path2, dpi=180); plt.close(fig)

    plot = recommendations.copy()
    plot["label"] = plot.trial_id.map(_display_label) + " · " + plot.scenario.map({
        "sensitivity": "Sensibilidad", "balanced": "Equilibrado"
    })
    metrics = ["smoke_recall", "fire_recall", "micro_precision", "micro_f1"]
    metric_labels = ["Recall humo", "Recall fuego", "Precisión", "F1"]
    x = np.arange(len(plot)); width = .18
    fig, ax = plt.subplots(figsize=(11, 5.7))
    metric_colors = ["#275D8C", "#B86A22", "#727B38", "#B55273"]
    for index, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        ax.bar(x + (index - 1.5) * width, plot[metric], width, label=label, color=color)
    ax.set(xticks=x, xticklabels=plot.label, ylim=(0, 1), ylabel="Métrica",
           title="Puntos operativos de los candidatos preliminares")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#D9D9D9", alpha=.6)
    ax.legend(frameon=False, ncol=4, loc="lower center")
    fig.tight_layout()
    path3 = figures / "03_operating_points.png"
    fig.savefig(path3, dpi=180); plt.close(fig)

    candidate_ids = list(grid.profile.drop_duplicates())
    fig, axes = plt.subplots(1, len(candidate_ids), figsize=(7 * len(candidate_ids), 5.5), squeeze=False)
    for ax, trial_id in zip(axes[0], candidate_ids):
        table = grid[grid.profile == trial_id].pivot(
            index="smoke_threshold", columns="fire_threshold", values="micro_f1"
        )
        image = ax.imshow(table.values, origin="lower", aspect="auto", vmin=.55, vmax=.85,
                          cmap="YlOrBr", extent=[table.columns.min(), table.columns.max(),
                                                  table.index.min(), table.index.max()])
        ax.set(title=_display_label(trial_id), xlabel="Umbral fuego", ylabel="Umbral humo")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="F1 micro", fraction=.025, pad=.03)
    fig.suptitle("Superficie de F1 por umbrales separados", fontsize=15)
    fig.subplots_adjust(top=.86, bottom=.13, left=.08, right=.90, wspace=.22)
    path4 = figures / "04_class_threshold_heatmaps.png"
    fig.savefig(path4, dpi=180); plt.close(fig)
    return [path, path2, path3, path4]


def markdown_table(table: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for record in table[columns].to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(formats[column].format(value) if column in formats else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_summary(output: Path, training: pd.DataFrame, candidates: pd.DataFrame,
                  recommendations: pd.DataFrame, winners: pd.DataFrame, config: dict) -> Path:
    training_view = training.copy()
    training_view["Configuración"] = training_view.trial_id.map(_display_label)
    candidate_view = candidates[candidates.selected_for_operational_evaluation].copy()
    candidate_view["Configuración"] = candidate_view.trial_id.map(_display_label)
    operating = recommendations.copy()
    operating["Configuración"] = operating.trial_id.map(_display_label)
    operating["Escenario"] = operating.scenario.map({"sensitivity": "Sensibilidad", "balanced": "Equilibrado"})
    winner_view = winners.copy()
    winner_view["Configuración"] = winner_view.trial_id.map(_display_label)
    winner_view["Escenario"] = winner_view.scenario.map({"sensitivity": "Sensibilidad", "balanced": "Equilibrado"})
    pct = {name: "{:.2%}" for name in ("smoke_recall", "fire_recall", "micro_precision", "micro_f1")}
    text = f"""# Evaluación del cribado de hiperparámetros YOLO26s

## Diseño

Se compararon el baseline y cuatro ensayos al mismo horizonte de
{config['screening_epochs']} épocas. Los dos candidatos preliminares se eligieron
por mAP50-95 de validación, con media de las últimas cinco épocas y recall como
desempates. Solo esos candidatos pasaron al barrido operativo por clase. El
conjunto de test no se cargó.

## Cribado de entrenamiento

{markdown_table(training_view.sort_values('best_map5095_50', ascending=False),
['Configuración', 'best_epoch_50', 'best_map5095_50', 'last5_mean_map5095',
 'last10_slope_map5095_per_epoch', 'delta_map5095_vs_baseline'],
{name: '{:.4f}' for name in ['best_map5095_50', 'last5_mean_map5095',
'last10_slope_map5095_per_epoch', 'delta_map5095_vs_baseline']})}

## Candidatos preliminares

{markdown_table(candidate_view, ['preliminary_rank', 'Configuración', 'best_map5095_50',
'last5_mean_map5095', 'recall_at_best_50'],
{name: '{:.4f}' for name in ['best_map5095_50', 'last5_mean_map5095', 'recall_at_best_50']})}

## Puntos operativos

Todos los puntos siguientes cumplen un máximo del {config['evaluation']['alarm_budget']:.0%}
de imágenes negativas con alarma.

{markdown_table(operating, ['Configuración', 'Escenario', 'smoke_threshold', 'fire_threshold',
'smoke_recall', 'fire_recall', 'micro_precision', 'micro_f1', 'negative_images_with_alarm'],
{**pct, 'smoke_threshold': '{:.2f}', 'fire_threshold': '{:.2f}'})}

## Recomendaciones provisionales

{markdown_table(winner_view, ['Escenario', 'Configuración', 'smoke_threshold', 'fire_threshold',
'smoke_recall', 'fire_recall', 'micro_precision', 'micro_f1', 'negative_images_with_alarm'],
{**pct, 'smoke_threshold': '{:.2f}', 'fire_threshold': '{:.2f}'})}

Estas recomendaciones son provisionales: la receta elegida debe entrenarse desde
cero durante 100 épocas y volver a superar el mismo protocolo. Una mejora
atribuida a los hiperparámetros requiere además semillas adicionales. No se ha
consultado test.
"""
    path = output / "RESUMEN_OPTIMIZACION_HIPERPARAMETROS.md"
    path.write_text(text, encoding="utf-8")
    return path


def preflight(config_path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    config = load_config(config_path)
    experiments = resolve_experiments(config, require_complete=False)
    rows = []
    for trial_id, experiment in experiments.items():
        rows.append({
            "trial_id": trial_id,
            "experiment_id": (config["baseline_experiment_id"] if trial_id == "baseline"
                              else config["trials"][trial_id]["experiment_id"]),
            "status": "absent" if experiment is None else experiment.get("status", "unknown"),
            "best_model_exists": bool(experiment and Path(experiment["best_model"]).exists()),
        })
    return pd.DataFrame(rows)


def run(config_path: Path = DEFAULT_CONFIG, *, offline: bool = False) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    raw_experiments = resolve_experiments(config, require_complete=True)
    experiments = {key: value for key, value in raw_experiments.items() if value is not None}
    training, curves = training_summary(experiments, config)
    candidates = select_preliminary(training, config["evaluation"]["candidate_count"])
    selected = candidates[candidates.selected_for_operational_evaluation]

    contract = pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])
    manifest = pipeline.rebased_manifest(contract)
    manifest = manifest.loc[manifest.split == "val"].copy()
    evaluation = dict(config["evaluation"])
    grids, cache_metadata = [], {}
    for row in selected.to_dict("records"):
        trial_id = row["trial_id"]
        experiment = experiments[trial_id]
        cache = get_prediction_cache(experiment, contract, evaluation, offline=offline)
        predictions_path = ROOT / cache["predictions_rel"]
        payloads = load_predictions(predictions_path, manifest, evaluation["prediction_confidence"])
        grid, _ = build_grid(trial_id, payloads, manifest, evaluation, evaluation)
        grids.append(grid)
        cache_metadata[trial_id] = cache
    grid = pd.concat(grids, ignore_index=True)
    recommendations = select_operating_points(grid, evaluation["alarm_budget"])
    winners = choose_overall_winners(recommendations)

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + fingerprint(config)[:8]
    output = OUTPUT_PARENT / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.yaml")
    training.to_csv(output / "training_screening.csv", index=False)
    curves.to_csv(output / "training_curves_50ep.csv", index=False)
    candidates.to_csv(output / "preliminary_candidates.csv", index=False)
    grid.to_csv(output / "class_threshold_grid.csv", index=False)
    recommendations.to_csv(output / "operating_recommendations.csv", index=False)
    winners.to_csv(output / "overall_winners.csv", index=False)
    figures = make_figures(training, curves, recommendations, grid, output)
    summary_path = write_summary(output, training, candidates, recommendations, winners, config)

    code_dir = output / "code"
    code_dir.mkdir()
    sources = [
        ROOT / "tools" / "run_hyperparameter_evaluation.py",
        ROOT / "tools" / "verify_hyperparameter_evaluation.py",
        ROOT / "tools" / "run_threshold_sweep.py",
        ROOT / "tools" / "run_class_threshold_and_fp_review.py",
        ROOT / "tfm_thresholds.py", ROOT / "tfm_evaluation.py", ROOT / "tfm_pipeline.py",
    ]
    for source in sources:
        if source.exists():
            shutil.copy2(source, code_dir / source.name)
    files = [
        output / "config.yaml", output / "training_screening.csv",
        output / "training_curves_50ep.csv", output / "preliminary_candidates.csv",
        output / "class_threshold_grid.csv", output / "operating_recommendations.csv",
        output / "overall_winners.csv", summary_path, *figures,
    ]
    metadata = {
        "schema_version": 1, "status": "complete", "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"], "split": "val",
        "test_inference_executed": False,
        "screening_experiments": {key: exp["experiment_id"] for key, exp in experiments.items()},
        "preliminary_candidates": selected[["trial_id", "experiment_id"]].to_dict("records"),
        "prediction_caches": cache_metadata,
        "files": {pipeline.project_relative(path, ROOT): pipeline.sha256_file(path) for path in files},
    }
    pipeline.write_json_atomic(output / "run.json", metadata)
    pointer = {
        "schema_version": 1, "status": "complete", "run_id": run_id,
        "output_rel": pipeline.project_relative(output, ROOT),
        "run_sha256": pipeline.sha256_file(output / "run.json"),
        "summary_rel": pipeline.project_relative(summary_path, ROOT),
        "summary_sha256": pipeline.sha256_file(summary_path),
    }
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    pipeline.write_json_atomic(OUTPUT_PARENT / "latest.json", pointer)
    print(f"Evaluación completa: {output}")
    print(winners[["scenario", "trial_id", "smoke_threshold", "fire_threshold",
                   "smoke_recall", "fire_recall", "micro_precision", "micro_f1",
                   "negative_images_with_alarm"]].to_string(index=False))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--offline", action="store_true", help="Exige cachés de predicción existentes.")
    parser.add_argument("--preflight", action="store_true", help="Muestra qué entrenamientos faltan.")
    args = parser.parse_args()
    if args.preflight:
        print(preflight(args.config).to_string(index=False))
    else:
        run(args.config, offline=args.offline)


if __name__ == "__main__":
    main()
