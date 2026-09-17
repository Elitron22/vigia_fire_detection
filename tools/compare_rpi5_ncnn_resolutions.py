"""Compara NCNN 768 y 640 con validación y benchmarks reales de Raspberry Pi 5."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline  # noqa: E402
from tfm_thresholds import evaluate_threshold, load_predictions  # noqa: E402


VARIANTS = {
    "ncnn_768": {
        "model": ROOT / "deployment/rpi5/model/yolo26s_768_ncnn_model",
        "predictions": ROOT / "artifacts/17_rpi5_deployment_validation/validation/ncnn_cache/20260915T232633Z_549e26/val_predictions.jsonl",
        "benchmark": ROOT / "artifacts/17_rpi5_deployment_validation/rpi5_actual/20260916T175620Z_ncnn/rpi5_ncnn_benchmark.json",
    },
    "ncnn_640": {
        "model": ROOT / "deployment/rpi5/model/yolo26s_640_ncnn_model",
        "predictions": ROOT / "artifacts/17_rpi5_deployment_validation/validation/ncnn640_cache/20260916T182255Z_93d24b/val_predictions.jsonl",
        "benchmark": ROOT / "artifacts/17_rpi5_deployment_validation/rpi5_actual/20260916T182840Z_ncnn640/rpi5_ncnn640_benchmark.json",
    },
}
OUTPUT_PARENT = ROOT / "artifacts/17_rpi5_deployment_validation/validation/ncnn_resolution_comparison"


def main() -> None:
    contract = pipeline.validate_prepared_dataset(ROOT, "dfire_seed42_val10_v1")
    staged = pipeline.stage_prepared_dataset(contract, workers=8)
    manifest = pipeline.rebased_manifest(contract, staged)
    val = manifest.loc[manifest.split == "val"].copy().reset_index(drop=True)
    records = {row["filename"]: row for row in val.to_dict("records")}
    eval_config = {"match_iou": 0.50, "size_area_boundaries": [0.01, 0.10]}

    results: dict[str, dict] = {}
    tabular_rows: list[dict] = []
    for name, paths in VARIANTS.items():
        calibration = json.loads((paths["model"] / "deployment_calibration.json").read_text(encoding="utf-8"))
        benchmark = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
        if calibration["status"] != "passed" or calibration["source_split"] != "val":
            raise AssertionError(f"Calibración no válida: {name}")
        if calibration["test_inference_executed"] is not False:
            raise AssertionError("La comparación no admite resultados procedentes de test.")
        selected = calibration["selected_ncnn"]
        payloads = load_predictions(paths["predictions"], val, 0.01)
        size_rows = []
        for class_name, threshold in (
            ("smoke", selected["smoke_threshold"]),
            ("fire", selected["fire_threshold"]),
        ):
            sizes = evaluate_threshold(payloads, records, threshold, eval_config, name)[2]
            size_rows.extend(sizes.loc[sizes.class_name.eq(class_name)].to_dict("records"))
        metrics = {
            key: selected[key] for key in (
                "smoke_threshold", "fire_threshold", "smoke_precision", "smoke_recall", "smoke_f1",
                "fire_precision", "fire_recall", "fire_f1", "micro_precision", "micro_recall",
                "micro_f1", "macro_recall", "minimum_class_recall", "negative_images_with_alarm",
                "negative_alarm_rate", "negative_image_fp_boxes",
            )
        }
        results[name] = {
            "validation_operating_metrics": metrics,
            "recall_by_size": size_rows,
            "rpi5_benchmark": {
                "latency_ms": benchmark["latency_ms"],
                "throughput_fps": benchmark["throughput_fps_from_mean"],
                "temperature_celsius": benchmark["temperature_celsius"],
                "throttled": benchmark["throttled"],
                "max_rss_kib": benchmark["max_rss_kib"],
            },
        }
        tabular_rows.append({"variant": name, **metrics, "latency_mean_ms": benchmark["latency_ms"]["mean"], "fps": benchmark["throughput_fps_from_mean"]})

    base = results["ncnn_768"]
    candidate = results["ncnn_640"]
    base_metrics = base["validation_operating_metrics"]
    candidate_metrics = candidate["validation_operating_metrics"]
    base_benchmark = base["rpi5_benchmark"]
    candidate_benchmark = candidate["rpi5_benchmark"]
    deltas = {
        f"{metric}_pp": 100.0 * (candidate_metrics[metric] - base_metrics[metric])
        for metric in (
            "smoke_precision", "smoke_recall", "smoke_f1", "fire_precision", "fire_recall",
            "fire_f1", "micro_precision", "micro_recall", "micro_f1", "macro_recall",
            "minimum_class_recall", "negative_alarm_rate",
        )
    }
    deltas.update({
        "latency_reduction_percent": 100.0 * (1.0 - candidate_benchmark["latency_ms"]["mean"] / base_benchmark["latency_ms"]["mean"]),
        "throughput_increase_percent": 100.0 * (candidate_benchmark["throughput_fps"] / base_benchmark["throughput_fps"] - 1.0),
        "speedup_factor": base_benchmark["latency_ms"]["mean"] / candidate_benchmark["latency_ms"]["mean"],
    })
    checks = {
        "validation_only": True,
        "test_inference_was_not_executed": True,
        "both_meet_negative_alarm_budget": all(results[name]["validation_operating_metrics"]["negative_alarm_rate"] <= 0.01 for name in results),
        "micro_f1_drop_at_most_1pp": deltas["micro_f1_pp"] >= -1.0,
        "fire_recall_drop_at_most_1_5pp": deltas["fire_recall_pp"] >= -1.5,
        "latency_reduction_at_least_25pct": deltas["latency_reduction_percent"] >= 25.0,
    }
    created = datetime.now(timezone.utc)
    output = OUTPUT_PARENT / created.strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(tabular_rows).to_csv(output / "comparison.csv", index=False)
    report = {
        "schema_version": 1,
        "created_utc": created.isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "source_split": "val",
        "test_inference_executed": False,
        "variants": results,
        "ncnn_640_minus_ncnn_768": deltas,
        "decision_checks": checks,
        "recommendation": "ncnn_640" if all(checks.values()) else "ncnn_768",
        "note": "Comparación operativa en validación y rendimiento real en Raspberry Pi 5; test permanece bloqueado.",
    }
    (output / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_PARENT / "latest.json").write_text(json.dumps({"report": str(output / "comparison_report.json")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
