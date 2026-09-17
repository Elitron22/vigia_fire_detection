"""Verificación independiente de los artefactos de comparación de resolución."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts" / "06_resolution_comparison" / "validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    latest = json.loads((PARENT / "latest.json").read_text(encoding="utf-8"))
    run_dir = ROOT / latest["run_rel"]
    summary_path = run_dir / "run_summary.json"
    assert sha256(summary_path) == latest["summary_sha256"], "latest.json no coincide"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["split"] == "val"
    assert summary["test_inference_executed"] is False
    for relative, expected in summary["output_hashes"].items():
        assert sha256(run_dir / relative) == expected, f"Hash incorrecto: {relative}"

    metrics = pd.read_csv(run_dir / "threshold_metrics.csv")
    scenarios = pd.read_csv(run_dir / "scenario_candidates.csv")
    standard = pd.read_csv(run_dir / "standard_metrics.csv")
    images = pd.read_csv(run_dir / "image_metrics.csv")
    boxes = pd.read_csv(run_dir / "selected_ground_truth.csv")
    paired = pd.read_csv(run_dir / "paired_comparison.csv")
    error_summary = pd.read_csv(run_dir / "error_review_summary.csv")
    error_examples = pd.read_csv(run_dir / "error_review_examples.csv")

    configurations = sorted(metrics.model_key.unique())
    thresholds = summary["config"]["thresholds"]
    expected_configurations = (
        len(summary["config"]["training_variants"])
        * len(summary["config"]["evaluation_sizes"])
    )
    assert len(configurations) == expected_configurations == 9
    assert len(metrics) == expected_configurations * len(thresholds) == 189
    assert not metrics.duplicated(["model_key", "threshold"]).any()
    assert len(standard) == expected_configurations * 3 and set(standard.scope) == {"all", "smoke", "fire"}
    assert len(images) == expected_configurations * len(thresholds) * 1721
    assert images.groupby(["model_key", "threshold"]).size().eq(1721).all()

    budget = float(summary["config"]["review_budget"])
    negatives = int(summary["negative_images"])
    exact_limit = math.floor(budget * negatives)
    chosen = scenarios[(scenarios.scenario == "alarm_02pct") & scenarios.feasible]
    assert len(chosen) == expected_configurations
    for row in chosen.itertuples():
        eligible = metrics[
            (metrics.model_key == row.model_key)
            & (metrics.negative_images_with_alarm <= exact_limit)
        ].sort_values(
            ["macro_recall", "minimum_class_recall", "micro_f1",
             "negative_images_with_alarm", "threshold"],
            ascending=[False, False, False, True, False],
        )
        expected = eligible.iloc[0]
        assert math.isclose(float(row.threshold), float(expected.threshold), abs_tol=1e-12)
        assert int(row.negative_images_with_alarm) <= exact_limit

    assert boxes.groupby("model_key").size().eq(2115).all()
    assert boxes[boxes.class_name == "fire"].groupby("model_key").size().eq(1155).all()
    assert set(error_summary.configuration) == set(configurations)
    assert len(error_examples) == 25 and error_examples.error_priority.is_monotonic_decreasing
    native_keys = ["train640_eval640", "train768_eval768", "train1024_eval1024"]
    native = boxes[boxes.model_key.isin(native_keys)]
    pivot = native[native.class_name == "fire"].pivot(
        index=["filename", "gt_index"], columns="model_key", values="status"
    )
    for candidate in native_keys[1:]:
        gains = ((pivot.train640_eval640 == "fn") & (pivot[candidate] == "tp")).sum()
        losses = ((pivot.train640_eval640 == "tp") & (pivot[candidate] == "fn")).sum()
        paired_boxes = paired[(paired.unit == "fire_box") & (paired.candidate_key == candidate)].iloc[0]
        assert int(paired_boxes.gains) == int(gains)
        assert int(paired_boxes.losses) == int(losses)

    print(f"Verificación correcta: {run_dir}")
    print(f"{expected_configurations} configuraciones × {len(thresholds)} umbrales; límite exacto: {exact_limit}/{negatives} negativas.")
    print("Test no consultado; hashes, selección operativa, cobertura y comparación emparejada coherentes.")


if __name__ == "__main__":
    main()
