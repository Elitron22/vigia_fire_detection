"""Evalúa en validación los detectores D-Fire ya entrenados, sin tocar test.

El script es reanudable: conserva evaluaciones completas y solo calcula lo que
falta. Es la versión automatizada del notebook 03 para comparar varios modelos.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tfm_pipeline as pipeline
from tfm_evaluation import (
    model_is_end_to_end,
    run_error_analysis,
    save_error_gallery,
    split_artifact_paths,
    write_summary_markdown,
)

DATASET_VERSION = "dfire_seed42_val10_v1"
MODEL_KEYS = ("yolov8n", "yolov8s", "yolo26n", "yolo26s")


def metric_array(metric, name: str) -> np.ndarray:
    value = getattr(metric, name, None)
    if value is None:
        return np.array([], dtype=float)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def standard_summary(metrics, model, experiment: dict, model_path: Path) -> dict:
    box = metrics.box
    precision, recall = metric_array(box, "p"), metric_array(box, "r")
    ap50, ap = metric_array(box, "ap50"), metric_array(box, "ap")
    rows = [{
        "scope": "all",
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }]
    for class_id, class_name in pipeline.CLASS_NAMES.items():
        rows.append({
            "scope": class_name,
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "mAP50": float(ap50[class_id]),
            "mAP50_95": float(ap[class_id]),
        })
    return {
        "experiment_id": experiment["experiment_id"],
        "model_key": experiment["model_key"],
        "model_path": str(model_path),
        "dataset_version": DATASET_VERSION,
        "split": "val",
        "end_to_end": model_is_end_to_end(model),
        "parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
        "weights_mib": model_path.stat().st_size / 1024**2,
        "metrics": rows,
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "evaluation_run_dir": str(Path(metrics.save_dir)),
    }


def latest_complete_error_summary(evaluation_root: Path) -> Path | None:
    candidates = sorted(
        (evaluation_root / "error_analysis").glob("*/val_error_summary.json"),
        reverse=True,
    )
    return candidates[0] if candidates else None


def evaluate_one(
    model_key: str,
    contract: dict,
    staged_dataset: Path,
    manifest: pd.DataFrame,
    *,
    run_error_analysis_flag: bool,
) -> tuple[dict, dict | None]:
    experiment = pipeline.resolve_experiment(model_key=model_key, root=PROJECT_ROOT)
    if experiment["dataset_version"] != DATASET_VERSION:
        raise ValueError(
            f"{experiment['experiment_id']} usa {experiment['dataset_version']}, no {DATASET_VERSION}."
        )
    model_path = Path(experiment["best_model"])
    evaluation_root = Path(experiment["experiment_root"]) / "evaluation" / "val"
    summary_path = evaluation_root / "evaluation_summary.json"
    metrics_path = evaluation_root / "metrics.csv"
    runtime_yaml = pipeline.write_runtime_data_yaml(
        contract, evaluation_root / "data_runtime.yaml", staged_dataset
    )

    print(f"\n=== {model_key}: {experiment['experiment_id']} ===", flush=True)
    model = None
    if summary_path.exists():
        summary = pipeline.read_json(summary_path)
        print("Evaluación estándar de val ya existente; se reutiliza.", flush=True)
    else:
        model = YOLO(str(model_path))
        pipeline.validate_class_mapping(model.names)
        device = 0 if torch.cuda.is_available() else "cpu"
        metrics = model.val(
            data=str(runtime_yaml),
            split="val",
            imgsz=640,
            batch=16,
            device=device,
            plots=True,
            project=str(evaluation_root),
            name="ultralytics",
            exist_ok=False,
            verbose=True,
        )
        summary = standard_summary(metrics, model, experiment, model_path)
        pipeline.write_json_atomic(summary_path, summary)
        pd.DataFrame(summary["metrics"]).to_csv(metrics_path, index=False)
        print(f"Guardado: {summary_path}", flush=True)

    error_summary = None
    if run_error_analysis_flag:
        existing_error = latest_complete_error_summary(evaluation_root)
        if existing_error:
            error_summary = pipeline.read_json(existing_error)
            print("Análisis operativo de val ya existente; se reutiliza.", flush=True)
        else:
            if model is None:
                model = YOLO(str(model_path))
                pipeline.validate_class_mapping(model.names)
            expected = manifest[manifest["split"] == "val"]
            error_output, error_table, error_summary = run_error_analysis(
                model,
                manifest,
                evaluation_root / "error_analysis",
                model_path=model_path,
                manifest_path=contract["manifest_path"],
                split="val",
                conf=.25,
                match_iou=.50,
                nms_iou=.70,
                imgsz=640,
                chunk_size=4,
                device=0 if torch.cuda.is_available() else "cpu",
                seed=42,
                ram_limit_gib=6.,
                expected_images=len(expected),
                expected_negatives=int((expected["box_count"] == 0).sum()),
            )
            write_summary_markdown(error_output, error_summary)
            paths = split_artifact_paths(error_output, "val")
            for mode, filename in (
                ("hardest", "hardest_examples.png"),
                ("negative_alarms", "negative_false_alarms.png"),
                ("random", "qualitative_predictions.png"),
            ):
                save_error_gallery(
                    error_table,
                    paths["predictions"],
                    error_output / filename,
                    count=12,
                    seed=42,
                    mode=mode,
                )
            print(f"Guardado: {paths['summary']}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, error_summary


def consolidate(results: list[tuple[dict, dict | None]]) -> Path:
    output = PROJECT_ROOT / "artifacts" / "04_model_comparison" / "validation"
    output.mkdir(parents=True, exist_ok=True)
    standard_rows, operating_rows, alarm_rows, index = [], [], [], []
    for standard, error in results:
        identity = {
            "experiment_id": standard["experiment_id"],
            "model_key": standard["model_key"],
            "end_to_end": standard["end_to_end"],
            "parameters": standard["parameters"],
            "weights_mib": standard["weights_mib"],
        }
        standard_rows.extend({**identity, **row} for row in standard["metrics"])
        if error:
            operating_rows.extend({**identity, **row} for row in error["box_metrics"])
            alarm_rows.extend({**identity, **row} for row in error["negative_image_alarms"])
        standard_path = Path(standard["evaluation_run_dir"]).parent / "evaluation_summary.json"
        error_path = Path(error["artifacts"]["summary"]) if error else None
        index.append({
            **identity,
            "standard_summary_rel": pipeline.project_relative(standard_path, PROJECT_ROOT),
            "error_summary_rel": (
                pipeline.project_relative(error_path, PROJECT_ROOT) if error_path else None
            ),
        })
    pd.DataFrame(standard_rows).to_csv(output / "validation_standard_metrics.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(output / "validation_operating_metrics.csv", index=False)
    pd.DataFrame(alarm_rows).to_csv(output / "validation_negative_alarm_rates.csv", index=False)
    pipeline.write_json_atomic(
        output / "evaluation_index.json",
        {
            "split": "val",
            "dataset_version": DATASET_VERSION,
            "operating_protocol": {
                "confidence": .25,
                "match_iou": .50,
                "requested_nms_iou": .70,
                "note": "NMS no se aplica a modelos end-to-end.",
            },
            "evaluations": index,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Omite el diagnóstico de FP/FN a confianza 0.25.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No hay CUDA: se aborta para no ejecutar cuatro evaluaciones en CPU.")
    contract = pipeline.validate_prepared_dataset(PROJECT_ROOT, DATASET_VERSION)
    staged_dataset = pipeline.stage_prepared_dataset(contract, workers=8)
    manifest = pipeline.rebased_manifest(contract, staged_dataset)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "models": args.models,
                "split": "val",
                "images": int((manifest["split"] == "val").sum()),
                "negative_images": int(
                    ((manifest["split"] == "val") & (manifest["box_count"] == 0)).sum()
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results = [
        evaluate_one(
            model_key,
            contract,
            staged_dataset,
            manifest,
            run_error_analysis_flag=not args.standard_only,
        )
        for model_key in args.models
    ]
    output = consolidate(results)
    print(f"\nComparación consolidada: {output}", flush=True)


if __name__ == "__main__":
    main()
