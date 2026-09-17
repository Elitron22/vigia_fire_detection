"""Calibra umbrales de la variante NCNN usando exclusivamente validación."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline  # noqa: E402
from tfm_thresholds import load_predictions  # noqa: E402
from tools.run_class_threshold_and_fp_review import build_grid  # noqa: E402


DEFAULT_MODEL = ROOT / "deployment" / "rpi5" / "model" / "yolo26s_768_ncnn_model"
DEFAULT_CACHE = (
    ROOT / "artifacts" / "17_rpi5_deployment_validation" / "validation" /
    "ncnn_cache" / "20260915T232633Z_549e26"
)
REFERENCE = (
    ROOT / "artifacts" / "12_final_validation_selection" / "validation" /
    "20260914T205147Z_65e09e1b" / "final_operating_points.csv"
)
OUTPUT_PARENT = (
    ROOT / "artifacts" / "17_rpi5_deployment_validation" / "validation" /
    "ncnn_calibration"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--variant-id", default="yolo26s_768_ncnn_rpi5")
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    parser.add_argument("--alarm-budget", type=float, default=0.01)
    parser.add_argument("--step", type=float, default=0.005)
    return parser.parse_args()


def metric_view(row: pd.Series) -> dict[str, float | int]:
    fields = [
        "smoke_threshold", "fire_threshold",
        "smoke_tp", "smoke_fp", "smoke_fn", "smoke_precision", "smoke_recall", "smoke_f1",
        "fire_tp", "fire_fp", "fire_fn", "fire_precision", "fire_recall", "fire_f1",
        "micro_tp", "micro_fp", "micro_fn", "micro_precision", "micro_recall", "micro_f1",
        "macro_recall", "minimum_class_recall", "negative_images",
        "negative_images_with_alarm", "negative_alarm_rate",
        "positive_image_fp_boxes", "negative_image_fp_boxes",
    ]
    result: dict[str, float | int] = {}
    integer_fields = {
        name for name in fields
        if name.endswith(("_tp", "_fp", "_fn")) or name in {
            "negative_images", "negative_images_with_alarm",
            "positive_image_fp_boxes", "negative_image_fp_boxes",
        }
    }
    for field in fields:
        value = row[field]
        result[field] = int(value) if field in integer_fields else float(value)
    return result


def main() -> None:
    args = parse_args()
    if not 0 < args.alarm_budget < 1 or not 0 < args.step <= 0.10:
        raise ValueError("Presupuesto o paso no válidos.")
    model = args.model.resolve()
    cache = args.cache.resolve()
    predictions = cache / "val_predictions.jsonl"
    if not model.is_dir() or not predictions.is_file():
        raise FileNotFoundError("Falta el modelo NCNN o su caché de validación.")

    contract = pipeline.validate_prepared_dataset(ROOT, "dfire_seed42_val10_v1")
    staged = pipeline.stage_prepared_dataset(contract, workers=8)
    manifest = pipeline.rebased_manifest(contract, staged)
    val = manifest.loc[manifest.split == "val"].copy().reset_index(drop=True)
    if len(val) != 1721 or int((val.box_count == 0).sum()) != 783:
        raise AssertionError("La población de validación no coincide con la congelada.")
    payloads = load_predictions(predictions, val, 0.01)

    count = int(round(0.99 / args.step))
    thresholds = [round(args.step * index, 4) for index in range(1, count + 1)]
    if thresholds[-1] < 0.99:
        thresholds.append(0.99)
    grid, _ = build_grid(
        args.variant_id,
        payloads,
        val,
        {"smoke_thresholds": thresholds, "fire_thresholds": thresholds},
        {"match_iou": 0.50, "size_area_boundaries": [0.01, 0.10]},
    )
    maximum_alarms = math.floor(args.alarm_budget * int(grid.iloc[0].negative_images) + 1e-12)
    feasible = grid.loc[grid.negative_images_with_alarm <= maximum_alarms]
    if feasible.empty:
        raise AssertionError("No existe un punto NCNN que cumpla el límite de alarmas.")
    selected = feasible.sort_values(
        ["minimum_class_recall", "macro_recall", "micro_f1", "micro_precision"],
        ascending=[False, False, False, False],
        kind="stable",
    ).iloc[0]
    original = grid.loc[
        grid.smoke_threshold.eq(0.36) & grid.fire_threshold.eq(0.16)
    ].iloc[0]
    reference = pd.read_csv(REFERENCE).loc[lambda frame: frame.candidate.eq("yolo26s")].iloc[0]

    selected_metrics = metric_view(selected)
    reference_metrics = metric_view(reference)
    original_metrics = metric_view(original)
    checks = {
        "validation_only": True,
        "test_inference_was_not_executed": True,
        "negative_alarm_rate_at_most_1pct": selected_metrics["negative_alarm_rate"] <= args.alarm_budget,
        "macro_recall_drop_vs_pytorch_at_most_0_5pp": (
            selected_metrics["macro_recall"] >= reference_metrics["macro_recall"] - 0.005
        ),
        "minimum_class_recall_drop_vs_pytorch_at_most_2pp": (
            selected_metrics["minimum_class_recall"]
            >= reference_metrics["minimum_class_recall"] - 0.02
        ),
        "micro_f1_drop_vs_pytorch_at_most_1pp": (
            selected_metrics["micro_f1"] >= reference_metrics["micro_f1"] - 0.01
        ),
    }
    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%SZ")
    output_parent = args.output_parent.resolve()
    output = output_parent / run_id
    output.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output / "threshold_grid.csv", index=False)
    pd.DataFrame([selected_metrics]).to_csv(output / "selected_operating_point.csv", index=False)

    report = {
        "schema_version": 1,
        "created_utc": created.isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "variant_id": args.variant_id,
        "role": "separate_edge_deployment_variant_not_final_test_model",
        "model_path": str(model),
        "source_split": "val",
        "test_inference_executed": False,
        "selection_policy": {
            "maximum_negative_alarm_rate": args.alarm_budget,
            "maximum_negative_images_with_alarm": maximum_alarms,
            "primary_metric": "minimum_class_recall",
            "tie_breakers": ["macro_recall", "micro_f1", "micro_precision"],
            "threshold_step": args.step,
        },
        "population": {"images": len(val), "negative_images": int((val.box_count == 0).sum())},
        "selected_ncnn": selected_metrics,
        "ncnn_at_frozen_pytorch_thresholds": original_metrics,
        "pytorch_reference": reference_metrics,
        "acceptance_checks": checks,
        "artifacts": {
            "prediction_cache": str(cache),
            "threshold_grid": str(output / "threshold_grid.csv"),
            "selected_operating_point": str(output / "selected_operating_point.csv"),
        },
        "note": (
            "Umbrales calibrados solo con validación. Esta variante no sustituye al modelo "
            "PyTorch congelado ni modifica ningún resultado de test."
        ),
    }
    report_path = output / "calibration_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (model / "deployment_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_parent / "latest.json").write_text(
        json.dumps({"run_id": run_id, "report": str(report_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
