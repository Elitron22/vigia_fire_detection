"""Compara todos los checkpoints D-Fire en validación con un 1 % de alarmas negativas."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline
from tfm_thresholds import evaluate_threshold, load_predictions
from tools.run_class_threshold_and_fp_review import build_grid
from tools.run_threshold_sweep import get_prediction_cache


DEFAULT_CONFIG = ROOT / "configs" / "all_models_01pct_comparison.yaml"
OUTPUT_PARENT = ROOT / "artifacts" / "13_all_models_01pct_comparison" / "validation"


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("split") != "val":
        raise ValueError("La comparación exige schema_version=1 y split=val.")
    if config.get("test_locked") is not True:
        raise ValueError("test_locked debe permanecer activado.")
    if not 0 < float(config["alarm_budget"]) < 1:
        raise ValueError("Presupuesto de alarmas no válido.")
    groups = {item["comparison_group"] for item in config["configurations"].values()}
    if groups != {"full_training", "hp_screening_50ep"}:
        raise ValueError("Los grupos de comparación no coinciden con el protocolo.")
    search = config["threshold_search"]
    if not (config["prediction_confidence"] <= search["minimum"] < search["maximum"] <= 1):
        raise ValueError("Intervalo de umbrales no válido.")
    return config


def threshold_values(config: dict) -> list[float]:
    search = config["threshold_search"]
    count = int(round((float(search["maximum"]) - float(search["minimum"])) / float(search["step"])))
    values = np.round(
        np.linspace(float(search["minimum"]), float(search["maximum"]), count + 1), 8
    ).tolist()
    if values[0] != float(search["minimum"]) or values[-1] != float(search["maximum"]):
        raise AssertionError("La malla no cubre los extremos declarados.")
    return values


def resolve_experiments(config: dict) -> dict[str, dict]:
    manifest_sha = pipeline.sha256_file(
        pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])["manifest_path"]
    )
    experiments: dict[str, dict] = {}
    for key, spec in config["configurations"].items():
        experiment = pipeline.resolve_experiment(experiment_id=spec["experiment_id"], root=ROOT)
        if experiment.get("status") != "complete":
            raise ValueError(f"Experimento no completo: {spec['experiment_id']}")
        if experiment.get("dataset_version") != config["dataset_version"]:
            raise ValueError(f"Dataset incompatible: {spec['experiment_id']}")
        descriptor_manifest_sha = experiment.get("dataset_manifest_sha256")
        if (descriptor_manifest_sha is not None and not pd.isna(descriptor_manifest_sha)
                and descriptor_manifest_sha != manifest_sha):
            raise ValueError(f"Manifiesto incompatible: {spec['experiment_id']}")
        if not Path(experiment["best_model"]).is_file():
            raise FileNotFoundError(experiment["best_model"])
        experiments[key] = experiment
    return experiments


def cache_for_configuration(
    key: str, spec: dict, experiment: dict, contract: dict, config: dict, offline: bool
) -> dict:
    eval_config = {
        "split": "val",
        "prediction_confidence": float(config["prediction_confidence"]),
        "match_iou": float(config["match_iou"]),
        "nms_iou": float(config["nms_iou"]),
        "imgsz": int(spec["eval_imgsz"]),
        "chunk_size": int(spec["chunk_size"]),
        "seed": int(config["seed"]),
        "ram_limit_gib": float(config["ram_limit_gib"]),
    }
    print(f"{key}: comprobando caché", flush=True)
    return get_prediction_cache(experiment, contract, eval_config, offline=offline)


def select_point(grid: pd.DataFrame, scenario: str, alarm_budget: float) -> pd.Series:
    feasible = grid[grid.negative_alarm_rate <= alarm_budget + 1e-12]
    if feasible.empty:
        raise ValueError("No existe un punto que cumpla el presupuesto de alarmas.")
    if scenario == "sensitivity":
        columns = ["macro_recall", "minimum_class_recall", "micro_f1", "micro_precision",
                   "negative_images_with_alarm", "smoke_threshold", "fire_threshold"]
        ascending = [False, False, False, False, True, False, False]
    elif scenario == "balanced":
        columns = ["micro_f1", "macro_recall", "micro_precision", "minimum_class_recall",
                   "negative_images_with_alarm", "smoke_threshold", "fire_threshold"]
        ascending = [False, False, False, False, True, False, False]
    else:
        raise ValueError(scenario)
    return feasible.sort_values(columns, ascending=ascending, kind="stable").iloc[0]


def standard_metrics_for(key: str, spec: dict, config: dict) -> pd.DataFrame:
    source_name = spec.get("standard_source")
    if source_name is None:
        return pd.DataFrame()
    source = pd.read_csv(ROOT / config["standard_sources"][source_name])
    selected = source[source[spec["standard_key_column"]].astype(str) == str(spec["standard_key"])].copy()
    if set(selected.scope) != {"all", "smoke", "fire"}:
        raise ValueError(f"Métricas estándar incompletas: {key}")
    keep = [column for column in (
        "scope", "precision", "recall", "mAP50", "mAP50_95", "parameters", "weights_mib",
        "preprocess_ms", "inference_ms", "postprocess_ms", "peak_cuda_allocated_gib"
    ) if column in selected]
    selected = selected[keep]
    selected.insert(0, "configuration", key)
    selected.insert(1, "label", spec["label"])
    selected.insert(2, "comparison_group", spec["comparison_group"])
    return selected


def compute(config: dict, offline: bool) -> tuple[pd.DataFrame, ...]:
    contract = pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])
    manifest = pipeline.rebased_manifest(contract)
    manifest = manifest[manifest.split == "val"].copy().reset_index(drop=True)
    records = {row["filename"]: row for row in manifest.to_dict("records")}
    experiments = resolve_experiments(config)
    thresholds = threshold_values(config)
    grid_config = {
        "prediction_confidence": float(config["prediction_confidence"]),
        "smoke_thresholds": thresholds,
        "fire_thresholds": thresholds,
    }
    operating_rows: list[dict] = []
    class_rows: list[dict] = []
    size_frames: list[pd.DataFrame] = []
    standard_frames: list[pd.DataFrame] = []
    grids: list[pd.DataFrame] = []
    cache_rows: list[dict] = []

    for key, spec in config["configurations"].items():
        experiment = experiments[key]
        cache = cache_for_configuration(key, spec, experiment, contract, config, offline)
        cache_rows.append({
            "configuration": key,
            "experiment_id": spec["experiment_id"],
            "eval_imgsz": int(spec["eval_imgsz"]),
            "predictions_rel": cache["predictions_rel"],
            "predictions_sha256": cache["predictions_sha256"],
            "checkpoint_sha256": cache["identity"]["checkpoint_sha256"],
        })
        payloads = load_predictions(
            ROOT / cache["predictions_rel"], manifest, float(config["prediction_confidence"])
        )
        eval_config = {
            "prediction_confidence": float(config["prediction_confidence"]),
            "match_iou": float(config["match_iou"]),
            "nms_iou": float(config["nms_iou"]),
            "imgsz": int(spec["eval_imgsz"]),
            "size_area_boundaries": config["size_area_boundaries"],
        }
        print(f"{key}: evaluando {len(thresholds) ** 2:,} pares de umbrales", flush=True)
        grid, _ = build_grid(key, payloads, manifest, grid_config, eval_config)
        grid = grid.rename(columns={"profile": "configuration"})
        grid.insert(1, "label", spec["label"])
        grid.insert(2, "comparison_group", spec["comparison_group"])
        grids.append(grid)

        selected_by_scenario: dict[str, pd.Series] = {}
        for scenario in ("sensitivity", "balanced"):
            selected = select_point(grid, scenario, float(config["alarm_budget"]))
            selected_by_scenario[scenario] = selected
            operating_rows.append({
                **selected.to_dict(),
                "configuration": key,
                "label": spec["label"],
                "experiment_id": spec["experiment_id"],
                "trained_imgsz": int(spec["trained_imgsz"]),
                "eval_imgsz": int(spec["eval_imgsz"]),
                "comparison_group": spec["comparison_group"],
                "scenario": scenario,
                "alarm_budget": float(config["alarm_budget"]),
            })

        sensitivity = selected_by_scenario["sensitivity"]
        for class_name in ("smoke", "fire"):
            class_rows.append({
                "configuration": key,
                "label": spec["label"],
                "comparison_group": spec["comparison_group"],
                "class_name": class_name,
                "threshold": float(sensitivity[f"{class_name}_threshold"]),
                **{metric: sensitivity[f"{class_name}_{metric}"]
                   for metric in ("tp", "fp", "fn", "precision", "recall", "f1")},
            })
            threshold = float(sensitivity[f"{class_name}_threshold"])
            _, _, sizes, _, _ = evaluate_threshold(
                payloads, records, threshold, eval_config, key
            )
            selected_sizes = sizes[sizes.class_name == class_name].copy()
            selected_sizes.insert(0, "configuration", key)
            selected_sizes.insert(1, "label", spec["label"])
            selected_sizes.insert(2, "comparison_group", spec["comparison_group"])
            size_frames.append(selected_sizes)

        standard = standard_metrics_for(key, spec, config)
        if not standard.empty:
            standard_frames.append(standard)

    operating = pd.DataFrame(operating_rows)
    operating = operating.drop(columns=["profile"], errors="ignore")
    classes = pd.DataFrame(class_rows)
    sizes = pd.concat(size_frames, ignore_index=True)
    standards = pd.concat(standard_frames, ignore_index=True, sort=False)
    threshold_grid = pd.concat(grids, ignore_index=True)
    caches = pd.DataFrame(cache_rows)
    return operating, classes, sizes, standards, threshold_grid, caches


def rank_operating(operating: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (group, scenario), table in operating.groupby(["comparison_group", "scenario"], sort=False):
        if scenario == "sensitivity":
            columns = ["macro_recall", "minimum_class_recall", "micro_f1", "micro_precision"]
        else:
            columns = ["micro_f1", "macro_recall", "micro_precision", "minimum_class_recall"]
        ranked = table.sort_values(columns, ascending=False, kind="stable").reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def build_figures(output: Path, ranked: pd.DataFrame, classes: pd.DataFrame) -> list[Path]:
    figure_dir = output / "figures"
    figure_dir.mkdir()
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    created = []

    full = ranked[(ranked.comparison_group == "full_training") & (ranked.scenario == "sensitivity")]
    full = full.sort_values("macro_recall")
    colors = [
        "#275D8C" if key == "yolo26s_640_eval640"
        else "#C58A1B" if key == "yolo26s_768_eval768"
        else "#B8C8D8"
        for key in full.configuration
    ]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    bars = ax.barh(full.label, full.macro_recall * 100, color=colors, edgecolor="#36516A", linewidth=.6)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in full.macro_recall * 100], padding=3, fontsize=8.5)
    ax.set(xlabel="Recall macro (%)", title="Comparación completa bajo ≤1 % de imágenes negativas con alarma")
    ax.set_xlim(0, max(86, float(full.macro_recall.max() * 100 + 5)))
    ax.grid(axis="x", color="#D9DEE3", linewidth=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = figure_dir / "01_full_training_macro_recall.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    full_classes = classes[classes.comparison_group == "full_training"].copy()
    order = full.sort_values("macro_recall", ascending=False).label.tolist()
    smoke = full_classes[full_classes.class_name == "smoke"].set_index("label").loc[order]
    fire = full_classes[full_classes.class_name == "fire"].set_index("label").loc[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hlines(y, smoke.recall * 100, fire.recall * 100, color="#A7ADB4", linewidth=1.2)
    ax.scatter(smoke.recall * 100, y, color="#275D8C", label="Humo", s=42, marker="o")
    ax.scatter(fire.recall * 100, y, facecolors="white", edgecolors="#B86A22", label="Fuego", s=48, marker="s", linewidth=1.5)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set(xlabel="Recall por clase (%)", title="Recall de humo y fuego en el punto de máxima sensibilidad")
    ax.set_xlim(50, 90)
    ax.grid(axis="x", color="#D9DEE3", linewidth=.7)
    ax.legend(loc="lower right", frameon=False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = figure_dir / "02_full_training_class_recall.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(full.micro_recall * 100, full.micro_precision * 100,
               c=["#275D8C" if key == "yolo26s_768_eval768" else "#C58A1B" for key in full.configuration],
               s=65, edgecolors="#34414C", linewidth=.6)
    annotation_offsets = {
        "yolo26s_640_eval640": (8, 8),
        "yolo26s_768_eval768": (8, 10),
        "yolo26s_768_eval640": (8, -15),
        "yolov8s_768_eval640": (-125, 10),
        "yolo26n_640_eval640": (8, -15),
    }
    for row in full.itertuples():
        if row.rank <= 5 or row.configuration in {"yolov8s_768_eval640", "yolo26s_768_eval768"}:
            ax.annotate(row.label, (row.micro_recall * 100, row.micro_precision * 100),
                        xytext=annotation_offsets.get(row.configuration, (5, 5)),
                        textcoords="offset points", fontsize=8)
    ax.set(xlabel="Recall micro (%)", ylabel="Precisión micro (%)",
           title="Intercambio entre sensibilidad y precisión con presupuesto del 1 %")
    ax.grid(color="#D9DEE3", linewidth=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = figure_dir / "03_full_training_precision_recall.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    hp = ranked[(ranked.comparison_group == "hp_screening_50ep") & (ranked.scenario == "sensitivity")]
    baseline = ranked[(ranked.configuration == "yolo26s_768_eval768") & (ranked.scenario == "sensitivity")].iloc[0]
    hp = hp.sort_values("macro_recall")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(hp.label, hp.macro_recall * 100, color="#D7C89B", edgecolor="#7A6840")
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in hp.macro_recall * 100], padding=3, fontsize=9)
    ax.axvline(baseline.macro_recall * 100, color="#275D8C", linestyle="--", linewidth=1.5,
               label=f"Baseline completo: {baseline.macro_recall * 100:.2f}%")
    ax.set(xlabel="Recall macro (%)", title="Ensayos de hiperparámetros a 50 épocas (grupo no equivalente)")
    ax.set_xlim(0, max(86, float(max(hp.macro_recall.max(), baseline.macro_recall) * 100 + 5)))
    ax.grid(axis="x", color="#D9DEE3", linewidth=.7)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = figure_dir / "04_hp_screening_macro_recall.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)
    return created


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Puesto | Configuración | Humo | Fuego | Precisión | Recall micro | F1 | Alarmas | Umbrales H/F |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {int(row.rank)} | {row.label} | {100*row.smoke_recall:.2f}% | "
            f"{100*row.fire_recall:.2f}% | {100*row.micro_precision:.2f}% | "
            f"{100*row.micro_recall:.2f}% | {100*row.micro_f1:.2f}% | "
            f"{int(row.negative_images_with_alarm)}/783 | "
            f"{row.smoke_threshold:.2f}/{row.fire_threshold:.2f} |"
        )
    return "\n".join(lines)


def write_summary(output: Path, ranked: pd.DataFrame, config: dict) -> None:
    full = ranked[(ranked.comparison_group == "full_training") & (ranked.scenario == "sensitivity")]
    hp = ranked[(ranked.comparison_group == "hp_screening_50ep") & (ranked.scenario == "sensitivity")]
    winner = full.sort_values("rank").iloc[0]
    prior = full[full.configuration == "yolov8s_768_eval640"].iloc[0]
    baseline = full[full.configuration == "yolo26s_768_eval768"].iloc[0]
    text = f"""# Comparación de todos los modelos con presupuesto de alarmas del 1 %

## Protocolo

- Partición: validación (`val`), 1.721 imágenes y 783 imágenes negativas.
- Test bloqueado: no se ha ejecutado inferencia sobre `test`.
- Predicciones conservadas desde confianza {config['prediction_confidence']:.2f}.
- Umbrales independientes para humo y fuego entre {config['threshold_search']['minimum']:.2f} y {config['threshold_search']['maximum']:.2f}, paso {config['threshold_search']['step']:.2f}.
- Restricción exacta: como máximo 7/783 imágenes negativas con alarma (0,894 % observado).
- Ranking principal: máximo recall macro; desempates por peor recall de clase, F1 y precisión.

## Resultado principal

El primer puesto es **{winner.label}**, con umbrales humo/fuego
**{winner.smoke_threshold:.2f}/{winner.fire_threshold:.2f}**, recall macro
**{100*winner.macro_recall:.2f} %**, precisión micro **{100*winner.micro_precision:.2f} %**,
recall micro **{100*winner.micro_recall:.2f} %** y F1 **{100*winner.micro_f1:.2f} %**.

La configuración elegida previamente, **{baseline.label}**, ocupa el puesto
**{int(baseline['rank'])}**. Frente a **{prior.label}**, conserva una ventaja de
**{100*(baseline.macro_recall-prior.macro_recall):.2f} puntos** de recall macro y
**{100*(baseline.micro_f1-prior.micro_f1):.2f} puntos** de F1 con el mismo máximo de alarmas.

## Modelos completos y configuraciones de resolución

{markdown_table(full.sort_values('rank'))}

## Ensayos de hiperparámetros a 50 épocas

Se presentan aparte porque no tienen el mismo horizonte de entrenamiento que los modelos
principales. Sirven como cribado, pero no pueden desplazar directamente a un entrenamiento
completo sin repetir la receta final.

{markdown_table(hp.sort_values('rank'))}

## Interpretación

- El presupuesto del 1 % modifica los umbrales operativos de cada configuración, no las métricas mAP estándar.
- La clasificación principal incluye todas las combinaciones de entrenamiento/inferencia ya estudiadas y los modelos nano.
- Los ensayos HP se incluyen por completitud, pero permanecen fuera de la selección final por su horizonte de 50 épocas.
- La comparación sigue siendo de una sola semilla por receta; las diferencias pequeñas no deben atribuirse únicamente a la arquitectura.
"""
    (output / "RESUMEN_TODOS_MODELOS_1PCT.md").write_text(text, encoding="utf-8")


def run(config_path: Path = DEFAULT_CONFIG, *, offline: bool = False) -> Path:
    config = load_config(config_path)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + fingerprint(config)[:8]
    output = OUTPUT_PARENT / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.yaml")
    metadata = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"],
        "split": "val",
        "test_inference_executed": False,
        "alarm_budget": float(config["alarm_budget"]),
    }
    pipeline.write_json_atomic(output / "run_summary.json", metadata)
    try:
        operating, classes, sizes, standards, grid, caches = compute(config, offline)
        ranked = rank_operating(operating)
        operating.to_csv(output / "operating_points.csv", index=False)
        ranked.to_csv(output / "ranked_operating_points.csv", index=False)
        classes.to_csv(output / "class_metrics_sensitivity.csv", index=False)
        sizes.to_csv(output / "size_metrics_sensitivity.csv", index=False)
        standards.to_csv(output / "standard_metrics.csv", index=False)
        grid.to_csv(output / "class_threshold_grid.csv", index=False)
        caches.to_csv(output / "prediction_sources.csv", index=False)
        pd.DataFrame(config.get("excluded_experiments", [])).to_csv(
            output / "excluded_experiments.csv", index=False
        )
        figures = build_figures(output, ranked, classes)
        write_summary(output, ranked, config)
        code_dir = output / "code"
        code_dir.mkdir()
        for relative in (
            "tools/run_all_models_01pct_comparison.py",
            "tools/verify_all_models_01pct_comparison.py",
            "tfm_thresholds.py", "tfm_evaluation.py", "tfm_pipeline.py",
        ):
            source = ROOT / relative
            if source.exists():
                shutil.copy2(source, code_dir / source.name)
        full = ranked[(ranked.comparison_group == "full_training") & (ranked.scenario == "sensitivity")]
        winner = full.sort_values("rank").iloc[0]
        metadata.update({
            "status": "complete",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "configurations": len(config["configurations"]),
            "full_training_configurations": int((operating.comparison_group == "full_training").sum() // 2),
            "hp_screening_configurations": int((operating.comparison_group == "hp_screening_50ep").sum() // 2),
            "thresholds_per_class": len(threshold_values(config)),
            "class_threshold_points": len(grid),
            "images": 1721,
            "negative_images": 783,
            "maximum_negative_images_with_alarm": 7,
            "selected_configuration": winner.configuration,
            "selected_label": winner.label,
            "selected_smoke_threshold": float(winner.smoke_threshold),
            "selected_fire_threshold": float(winner.fire_threshold),
            "selected_macro_recall": float(winner.macro_recall),
            "figures": [path.relative_to(output).as_posix() for path in figures],
        })
        metadata["output_hashes"] = {
            path.relative_to(output).as_posix(): pipeline.sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "run_summary.json"
        }
        pipeline.write_json_atomic(output / "run_summary.json", metadata)
        pipeline.write_json_atomic(OUTPUT_PARENT / "latest.json", {
            "run_id": run_id,
            "run_rel": pipeline.project_relative(output, ROOT),
            "summary_sha256": pipeline.sha256_file(output / "run_summary.json"),
        })
        print(f"Comparación completa: {output}", flush=True)
        print(full.sort_values("rank")[[
            "rank", "label", "smoke_threshold", "fire_threshold", "macro_recall",
            "micro_precision", "micro_f1", "negative_images_with_alarm"
        ]].to_string(index=False), flush=True)
        return output
    except BaseException as exc:
        metadata.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        pipeline.write_json_atomic(output / "run_summary.json", metadata)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--offline", action="store_true", help="Exige que todas las cachés existan.")
    args = parser.parse_args()
    run(args.config, offline=args.offline)


if __name__ == "__main__":
    main()
