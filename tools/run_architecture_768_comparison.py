"""Compara YOLOv8s y YOLO26s entrenados a 768 sobre validación D-Fire."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
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
from tfm_thresholds import choose_scenarios, evaluate_threshold, load_predictions, sweep_model
from tools.run_resolution_comparison import image_detection_metrics, standard_validation, threshold_config
from tools.run_threshold_sweep import get_prediction_cache


COLORS = {"yolov8s_768": "#275D8C", "yolo26s_768": "#D95F35"}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("split") != "val":
        raise ValueError("La comparación requiere schema_version=1 y split=val.")
    candidates = config.get("candidates", {})
    if set(candidates) != {"yolov8s_768", "yolo26s_768"}:
        raise ValueError("Se requieren exactamente yolov8s_768 y yolo26s_768.")
    if config.get("baseline") not in candidates or config.get("candidate") not in candidates:
        raise ValueError("Baseline o candidato no declarados.")
    if sorted(config.get("evaluation_sizes", [])) != [640, 768]:
        raise ValueError("La evaluación debe ejecutarse a 640 y 768.")
    thresholds = np.asarray(config["thresholds"], dtype=float)
    if np.any(np.diff(thresholds) <= 0) or config["prediction_confidence"] not in thresholds:
        raise ValueError("La malla de umbrales no es válida.")
    return config


def exact_mcnemar(a, b) -> dict[str, float | int]:
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


def training_resources(experiments: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, experiment in experiments.items():
        results = pd.read_csv(Path(experiment["training_run_dir"]) / "results.csv")
        best_index = int(results["metrics/mAP50-95(B)"].idxmax())
        rows.append({
            "configuration": key,
            "experiment_id": experiment["experiment_id"],
            "epochs_completed": int(results.epoch.iloc[-1]),
            "best_epoch": int(results.epoch.iloc[best_index]),
            "training_seconds": float(results.time.iloc[-1]),
            "training_hours": float(results.time.iloc[-1] / 3600),
            "best_training_map50_95": float(results["metrics/mAP50-95(B)"].iloc[best_index]),
            "weights_mib": Path(experiment["best_model"]).stat().st_size / 1024**2,
        })
    return pd.DataFrame(rows)


def selected_details(
    scenarios: pd.DataFrame,
    payloads: dict[str, list[dict]],
    records: dict[str, dict],
    config: dict,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    chosen = scenarios[(scenarios.scenario == "alarm_02pct") & scenarios.feasible].copy()
    expected = len(config["candidates"]) * len(config["evaluation_sizes"])
    if len(chosen) != expected:
        raise AssertionError(f"Faltan puntos operativos: {len(chosen)}/{expected}.")
    image_tables: dict[str, pd.DataFrame] = {}
    gt_tables: dict[str, pd.DataFrame] = {}
    image_rows = []
    for row in chosen.itertuples():
        combo_config = threshold_config(config, int(row.eval_imgsz))
        _, images, _, gt, _ = evaluate_threshold(
            payloads[row.model_key], records, float(row.threshold), combo_config, row.model_key, details=True
        )
        image_tables[row.model_key] = images
        gt_tables[row.model_key] = gt
        image_rows.append({"configuration": row.model_key, **image_detection_metrics(images)})
    return chosen, image_tables, gt_tables, pd.DataFrame(image_rows)


def paired_comparisons(
    image_tables: dict[str, pd.DataFrame],
    gt_tables: dict[str, pd.DataFrame],
    config: dict,
) -> pd.DataFrame:
    rows = []
    for eval_imgsz in config["evaluation_sizes"]:
        baseline_key = f"{config['baseline']}_eval{eval_imgsz}"
        candidate_key = f"{config['candidate']}_eval{eval_imgsz}"
        old_images = image_tables[baseline_key].set_index("filename").sort_index()
        new_images = image_tables[candidate_key].set_index("filename").sort_index()
        if not old_images.index.equals(new_images.index):
            raise AssertionError("Las configuraciones no contienen las mismas imágenes.")
        for class_name in ("smoke", "fire"):
            positive = old_images[f"{class_name}_gt"] > 0
            stats = exact_mcnemar(
                old_images.loc[positive, f"{class_name}_tp"] > 0,
                new_images.loc[positive, f"{class_name}_tp"] > 0,
            )
            rows.append({
                "eval_imgsz": eval_imgsz, "class_name": class_name, "unit": "positive_image",
                "n": int(positive.sum()),
                "baseline_rate": float((old_images.loc[positive, f"{class_name}_tp"] > 0).mean()),
                "candidate_rate": float((new_images.loc[positive, f"{class_name}_tp"] > 0).mean()),
                **stats,
            })
            keys = ["filename", "gt_index", "class_name"]
            old_gt = gt_tables[baseline_key]
            new_gt = gt_tables[candidate_key]
            paired = old_gt[keys + ["status"]].merge(
                new_gt[keys + ["status"]], on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one"
            )
            paired = paired[paired.class_name == class_name]
            stats = exact_mcnemar(paired.status_baseline == "tp", paired.status_candidate == "tp")
            rows.append({
                "eval_imgsz": eval_imgsz, "class_name": class_name, "unit": "box", "n": len(paired),
                "baseline_rate": float((paired.status_baseline == "tp").mean()),
                "candidate_rate": float((paired.status_candidate == "tp").mean()),
                **stats,
            })
    return pd.DataFrame(rows)


def error_summary(chosen: pd.DataFrame, image_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for selected in chosen.itertuples():
        images = image_tables[selected.model_key]
        negative = images.gt_count == 0
        rows.append({
            "configuration": selected.model_key,
            "threshold": selected.threshold,
            "smoke_fp": int(images.smoke_fp.sum()), "smoke_fn": int(images.smoke_fn.sum()),
            "fire_fp": int(images.fire_fp.sum()), "fire_fn": int(images.fire_fn.sum()),
            "positive_image_fp_boxes": int((images.loc[~negative, "smoke_fp"] + images.loc[~negative, "fire_fp"]).sum()),
            "negative_image_fp_boxes": int((images.loc[negative, "smoke_fp"] + images.loc[negative, "fire_fp"]).sum()),
            "negative_images_with_alarm": int(((images.loc[negative, "smoke_fp"] + images.loc[negative, "fire_fp"]) > 0).sum()),
        })
    return pd.DataFrame(rows)


def build_figures(metrics: pd.DataFrame, chosen: pd.DataFrame, config: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figures = output / "figures"
    figures.mkdir()
    labels = {key: value["label"] for key, value in config["candidates"].items()}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title) in zip(axes.flat, [
        ("smoke_recall", "Recall de humo"), ("fire_recall", "Recall de fuego"),
        ("micro_f1", "F1 micro"), ("negative_alarm_rate", "Negativas con alarma"),
    ]):
        for candidate in config["candidates"]:
            for eval_imgsz, style in ((640, "-"), (768, "--")):
                key = f"{candidate}_eval{eval_imgsz}"
                table = metrics[metrics.model_key == key].sort_values("threshold")
                ax.plot(table.threshold, table[metric], color=COLORS[candidate], linestyle=style,
                        label=f"{labels[candidate]} · eval {eval_imgsz}")
        ax.set(title=title, xlabel="Umbral", xlim=(0, 1), ylim=(0, 1))
        if metric == "negative_alarm_rate":
            ax.set_ylim(0, max(0.03, float(metrics[metric].max()) * 1.12))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="#E1E4E6")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Comparación de arquitecturas entrenadas a 768", fontsize=17, y=.985)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .945), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(figures / "01_threshold_curves.png", dpi=170)
    fig.savefig(figures / "01_threshold_curves.svg")
    plt.close(fig)

    focus = chosen[chosen.eval_imgsz == config["focus_evaluation_size"]].set_index("model_key")
    order = [f"{key}_eval{config['focus_evaluation_size']}" for key in config["candidates"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    columns = ["smoke_recall", "fire_recall", "micro_precision", "micro_f1"]
    x = np.arange(len(columns)); width = .34
    for index, key in enumerate(order):
        candidate = key.rsplit("_eval", 1)[0]
        ax.bar(x + (index - .5) * width, focus.loc[key, columns], width,
               label=labels[candidate], color=COLORS[candidate])
    ax.set_xticks(x, ["Recall humo", "Recall fuego", "Precisión micro", "F1 micro"])
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y", color="#E1E4E6")
    ax.legend(frameon=False); ax.set_title("Puntos operativos a 640 · límite del 2 % de falsas alarmas")
    fig.tight_layout(); fig.savefig(figures / "02_operating_points_640.png", dpi=170)
    fig.savefig(figures / "02_operating_points_640.svg"); plt.close(fig)


def write_summary(
    output: Path, chosen: pd.DataFrame, standard: pd.DataFrame, image_metrics: pd.DataFrame,
    paired: pd.DataFrame, resources: pd.DataFrame, config: dict,
) -> None:
    focus_size = config["focus_evaluation_size"]
    focus = chosen[chosen.eval_imgsz == focus_size].set_index("model_key")
    image_focus = image_metrics.set_index("configuration")
    lines = [
        "# Comparación YOLOv8s frente a YOLO26s entrenados a 768", "",
        "Evaluación cerrada sobre validación D-Fire. No se ha consultado test.", "",
        f"## Punto operativo principal: inferencia a {focus_size}", "",
        "| Modelo | Umbral | Recall humo | Recall fuego | Precisión micro | F1 micro | Recall imágenes fuego | Alarmas negativas |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in config["candidates"]:
        key = f"{candidate}_eval{focus_size}"
        row = focus.loc[key]
        image_row = image_focus.loc[key]
        lines.append(
            f"| {config['candidates'][candidate]['label']} | {row.threshold:.2f} | {row.smoke_recall:.2%} | "
            f"{row.fire_recall:.2%} | {row.micro_precision:.2%} | {row.micro_f1:.2%} | "
            f"{image_row.fire_image_recall:.2%} | {int(row.negative_images_with_alarm)}/{int(row.negative_images)} |"
        )
    lines += ["", "## Comparación emparejada", ""]
    for row in paired[paired.eval_imgsz == focus_size].itertuples():
        unit = "imágenes positivas" if row.unit == "positive_image" else "cajas"
        lines.append(
            f"- {row.class_name}, {unit}: {row.gains} recuperaciones y {row.losses} pérdidas al pasar a YOLO26s; "
            f"p exacta={row.p_value:.4g}, n={row.n}."
        )
    lines += ["", "## Entrenamiento y validación estándar", ""]
    for row in resources.itertuples():
        lines.append(
            f"- {row.configuration}: {row.training_hours:.2f} h, mejor época {row.best_epoch}, "
            f"mejor mAP50-95 durante entrenamiento {row.best_training_map50_95:.4f}, pesos {row.weights_mib:.1f} MiB."
        )
    lines += ["", "Las métricas estándar de Ultralytics y sus tiempos están en `standard_metrics.csv`.", "",
              "## Alcance", "", "La comparación usa las mismas 1.721 imágenes y anotaciones, una sola semilla por arquitectura, "
              "IoU de acierto 0,50 y una confianza común para humo y fuego. Las diferencias pequeñas deben confirmarse con "
              "semillas adicionales antes de atribuirlas a la arquitectura.", ""]
    (output / "RESUMEN_ARQUITECTURAS_768.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    contract = pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])
    staged = pipeline.stage_prepared_dataset(contract, workers=int(config["staging_workers"]))
    manifest = pipeline.rebased_manifest(contract, staged)
    val_manifest = manifest[manifest.split == "val"].copy().reset_index(drop=True)
    records = {row["filename"]: row for row in val_manifest.to_dict("records")}
    experiments = {}
    for key, spec in config["candidates"].items():
        experiment = pipeline.resolve_experiment(experiment_id=spec["experiment_id"], root=ROOT)
        if experiment["status"] != "complete" or experiment["dataset_version"] != config["dataset_version"]:
            raise ValueError(f"Experimento no utilizable: {spec['experiment_id']}")
        if int(experiment["train_config"]["imgsz"]) != int(spec["trained_imgsz"]):
            raise ValueError(f"Resolución incoherente: {spec['experiment_id']}")
        experiments[key] = experiment

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + fingerprint(config)[:8]
    parent = ROOT / "artifacts" / "09_architecture_768_comparison" / "validation"
    output = parent / run_id
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    code_dir = output / "code"; code_dir.mkdir()
    for path in (Path(__file__), ROOT / "tfm_thresholds.py", ROOT / "tfm_evaluation.py", ROOT / "tfm_pipeline.py"):
        shutil.copy2(path, code_dir / path.name)
    summary = {
        "schema_version": 1, "status": "running", "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "split": "val",
        "dataset_version": config["dataset_version"], "test_inference_executed": False,
        "manifest_sha256": pipeline.sha256_file(contract["manifest_path"]), "config": config,
        "environment": pipeline.environment_snapshot(),
    }
    pipeline.write_json_atomic(output / "run_summary.json", summary)

    standard_rows, metric_frames, size_frames = [], [], []
    payloads: dict[str, list[dict]] = {}
    try:
        for candidate, experiment in experiments.items():
            for eval_imgsz in config["evaluation_sizes"]:
                key = f"{candidate}_eval{eval_imgsz}"
                print(f"\n=== {key} ===", flush=True)
                standard_result = standard_validation(experiment, contract, staged, config, eval_imgsz)
                for metric in standard_result["metrics"]:
                    standard_rows.append({
                        "configuration": key, "eval_imgsz": eval_imgsz, **metric,
                        **{f"{name}_ms": value for name, value in standard_result["speed_ms_per_image"].items()},
                        "peak_cuda_allocated_gib": standard_result["peak_cuda_allocated_gib"],
                    })
                combo_config = threshold_config(config, eval_imgsz)
                cache = get_prediction_cache(experiment, contract, combo_config)
                current_payloads = load_predictions(ROOT / cache["predictions_rel"], val_manifest, config["prediction_confidence"])
                payloads[key] = current_payloads
                metrics, _, sizes = sweep_model(current_payloads, val_manifest, combo_config, key)
                metrics["architecture"] = candidate; metrics["eval_imgsz"] = eval_imgsz
                sizes["architecture"] = candidate; sizes["eval_imgsz"] = eval_imgsz
                metric_frames.append(metrics); size_frames.append(sizes)

        standard = pd.DataFrame(standard_rows)
        metrics = pd.concat(metric_frames, ignore_index=True)
        sizes = pd.concat(size_frames, ignore_index=True)
        scenarios = choose_scenarios(metrics, config)
        scenarios["eval_imgsz"] = scenarios.model_key.str.extract(r"_eval(\d+)$").astype(int)
        chosen, image_tables, gt_tables, image_metrics = selected_details(scenarios, payloads, records, config)
        paired = paired_comparisons(image_tables, gt_tables, config)
        errors = error_summary(chosen, image_tables)
        resources = training_resources(experiments)

        standard.to_csv(output / "standard_metrics.csv", index=False)
        metrics.to_csv(output / "threshold_metrics.csv", index=False)
        scenarios.to_csv(output / "scenario_candidates.csv", index=False)
        chosen.to_csv(output / "selected_operating_points.csv", index=False)
        sizes.to_csv(output / "size_metrics.csv", index=False)
        image_metrics.to_csv(output / "image_detection_metrics.csv", index=False)
        paired.to_csv(output / "paired_comparison.csv", index=False)
        errors.to_csv(output / "selected_error_summary.csv", index=False)
        resources.to_csv(output / "training_resources.csv", index=False)
        build_figures(metrics, chosen, config, output)
        write_summary(output, chosen, standard, image_metrics, paired, resources, config)

        if len(metrics) != len(config["candidates"]) * len(config["evaluation_sizes"]) * len(config["thresholds"]):
            raise AssertionError("Número de puntos del barrido incorrecto.")
        if set(metrics.images) != {len(val_manifest)} or set(metrics.negative_images) != {int((val_manifest.box_count == 0).sum())}:
            raise AssertionError("Cobertura de validación incorrecta.")
        if pipeline.sha256_file(contract["manifest_path"]) != summary["manifest_sha256"]:
            raise AssertionError("El manifiesto cambió durante la evaluación.")
        summary.update(
            status="complete", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            configurations=len(config["candidates"]) * len(config["evaluation_sizes"]),
            thresholds=len(config["thresholds"]), images_per_configuration=len(val_manifest),
            negative_images=int((val_manifest.box_count == 0).sum()),
            output_hashes={str(path.relative_to(output)): pipeline.sha256_file(path)
                           for path in output.rglob("*") if path.is_file() and path.name != "run_summary.json"},
        )
        pipeline.write_json_atomic(output / "run_summary.json", summary)
        pipeline.write_json_atomic(parent / "latest.json", {
            "run_id": run_id, "run_rel": pipeline.project_relative(output, ROOT),
            "summary_sha256": pipeline.sha256_file(output / "run_summary.json"),
        })
        print(f"\nComparación completa: {output}", flush=True)
        print(chosen[["model_key", "threshold", "smoke_recall", "fire_recall", "micro_precision", "micro_f1", "negative_images_with_alarm"]].to_string(index=False))
        return output
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        pipeline.write_json_atomic(output / "run_summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "architecture_768_comparison.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
