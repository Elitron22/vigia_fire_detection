"""Valida en `val` que el NCNN conserva el punto operativo final del TFM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfm_evaluation import run_error_analysis, split_artifact_paths  # noqa: E402
from tfm_thresholds import load_predictions  # noqa: E402
import tfm_pipeline as pipeline  # noqa: E402
from tools.run_class_threshold_and_fp_review import build_grid  # noqa: E402


DEFAULT_MODEL = ROOT / "deployment" / "rpi5" / "model" / "yolo26s_768_ncnn_model"
DATASET_MANIFEST = ROOT / "artifacts" / "datasets" / "dfire_seed42_val10_v1" / "dataset_manifest.csv"
REFERENCE = ROOT / "artifacts" / "12_final_validation_selection" / "validation" / "20260914T205147Z_65e09e1b" / "final_operating_points.csv"
CACHE_PARENT = ROOT / "artifacts" / "17_rpi5_deployment_validation" / "validation" / "ncnn_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--cache-parent", type=Path, default=CACHE_PARENT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    contract = pipeline.validate_prepared_dataset(ROOT, "dfire_seed42_val10_v1")
    staged = pipeline.stage_prepared_dataset(contract, workers=8)
    manifest = pipeline.rebased_manifest(contract, staged)
    val = manifest.loc[manifest.split == "val"].copy().reset_index(drop=True)
    model = YOLO(str(model_path))
    cache, _, _ = run_error_analysis(
        model,
        manifest,
        args.cache_parent.resolve(),
        manifest_path=DATASET_MANIFEST,
        split="val",
        conf=0.01,
        match_iou=0.50,
        nms_iou=0.70,
        imgsz=args.imgsz,
        chunk_size=1,
        device="cpu",
        seed=42,
        ram_limit_gib=8.0,
        expected_images=1721,
        expected_negatives=783,
    )
    predictions = split_artifact_paths(cache, "val")["predictions"]
    payloads = load_predictions(predictions, val, 0.01)
    grid_config = {
        "smoke_thresholds": [0.36],
        "fire_thresholds": [0.16],
    }
    eval_config = {"match_iou": 0.50, "size_area_boundaries": [0.01, 0.10]}
    grid, _ = build_grid("yolo26s_ncnn", payloads, val, grid_config, eval_config)
    deployed = grid.iloc[0].to_dict()
    reference_table = pd.read_csv(REFERENCE)
    reference = reference_table.loc[reference_table.candidate == "yolo26s"].iloc[0].to_dict()

    comparisons = {}
    for metric in (
        "smoke_precision", "smoke_recall", "smoke_f1",
        "fire_precision", "fire_recall", "fire_f1",
        "micro_precision", "micro_recall", "micro_f1", "macro_recall",
        "negative_alarm_rate",
    ):
        comparisons[metric] = {
            "pytorch": float(reference[metric]),
            "ncnn": float(deployed[metric]),
            "delta": float(deployed[metric]) - float(reference[metric]),
        }
    checks = {
        "negative_alarm_rate_at_most_1pct": float(deployed["negative_alarm_rate"]) <= 0.01,
        "smoke_recall_drop_at_most_1pp": float(deployed["smoke_recall"]) >= float(reference["smoke_recall"]) - 0.01,
        "fire_recall_drop_at_most_1pp": float(deployed["fire_recall"]) >= float(reference["fire_recall"]) - 0.01,
        "micro_f1_drop_at_most_1pp": float(deployed["micro_f1"]) >= float(reference["micro_f1"]) - 0.01,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "split": "val",
        "test_inference_executed": False,
        "model_path": str(model_path),
        "operating_point": {"imgsz": args.imgsz, "smoke_threshold": 0.36, "fire_threshold": 0.16},
        "images": 1721,
        "negative_images": 783,
        "ncnn_metrics": deployed,
        "comparisons": comparisons,
        "acceptance_checks": checks,
        "cache_path": str(cache),
        "note": "Comprobación de portabilidad; no selecciona modelo ni reajusta umbrales.",
    }
    report_path = model_path / "deployment_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
