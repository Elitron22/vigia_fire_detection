"""Verifica los 63 puntos desde las predicciones, sin llamar al motor del barrido."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tfm_pipeline as pipeline


def verify(output=None, *, record=False):
    parent = ROOT / "artifacts" / "05_threshold_sweep" / "validation"
    output = Path(output) if output else ROOT / pipeline.read_json(parent / "latest.json")["run_rel"]
    summary_path = output / "run_summary.json"
    summary = pipeline.read_json(summary_path)
    assert summary["status"] == "complete" and summary["split"] == "val"
    assert summary["test_inference_executed"] is False
    config = summary["config"]
    manifest = pd.read_csv(ROOT / summary["manifest_rel"])
    assert pipeline.sha256_file(ROOT / summary["manifest_rel"]) == summary["manifest_sha256"]
    assert pipeline.sha256_file(ROOT / summary["historical_index_rel"]) == summary["historical_index_sha256"]
    validation = manifest[manifest.split == "val"].set_index("filename")
    assert len(validation) == 1721 and (validation.box_count == 0).sum() == 783
    assert validation.smoke_boxes.sum() == 960 and validation.fire_boxes.sum() == 1155
    metrics = pd.read_csv(output / "threshold_metrics.csv")
    images = pd.read_csv(output / "image_metrics.csv")
    sizes = pd.read_csv(output / "size_metrics.csv")
    candidates = pd.read_csv(output / "scenario_candidates.csv")
    assert not metrics[["model_key", "threshold"]].duplicated().any()
    assert not images[["model_key", "threshold", "filename"]].duplicated().any()
    assert len(metrics) == len(config["experiments"]) * len(config["thresholds"])
    verified_matches, verified_rows, verified_source_images = 0, 0, 0
    for model_key, cache in summary["inputs"].items():
        for key in ("summary", "predictions"):
            assert pipeline.sha256_file(ROOT / cache[f"{key}_rel"]) == cache[f"{key}_sha256"]
        cache_summary = pipeline.read_json(ROOT / cache["summary_rel"])
        assert cache_summary["status"] == "complete" and cache_summary["split"] == "val"
        exp = pipeline.resolve_experiment(experiment_id=config["experiments"][model_key], root=ROOT)
        assert pipeline.sha256_file(Path(exp["best_model"])) == cache["identity"]["checkpoint_sha256"]
        payloads = [json.loads(line) for line in (ROOT / cache["predictions_rel"]).open()]
        assert len(payloads) == len(validation) and {p["filename"] for p in payloads} == set(validation.index)
        for payload in payloads:
            used = set()
            for prediction in payload["predictions"]:
                if prediction["status"] != "tp":
                    continue
                index = prediction["matched_gt_index"]
                assert index not in used
                used.add(index)
                truth = payload["ground_truth"][index]
                assert prediction["class_id"] == truth["class_id"]
                a, b = np.array(prediction["xyxy"]), np.array(truth["xyxy"])
                intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
                union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
                assert intersection / union >= config["match_iou"] - 1e-8
                verified_matches += 1
        model_metrics = metrics[metrics.model_key == model_key]
        assert set(model_metrics.threshold) == set(config["thresholds"])
        for threshold in config["thresholds"]:
            selected = images[(images.model_key == model_key) & np.isclose(images.threshold, threshold)].set_index("filename")
            assert set(selected.index) == set(validation.index)
            counts = {"smoke": {"tp": 0, "fp": 0, "fn": 0}, "fire": {"tp": 0, "fp": 0, "fn": 0}}
            alarms, negative_boxes = 0, 0
            size_counts = {(cls, size): [0, 0] for cls in ("smoke", "fire") for size in ("small", "medium", "large")}
            for payload in payloads:
                # El matching a confianza mínima se hizo en la pasada de inferencia.
                # Filtrar su secuencia ordenada elimina solo candidatos de menor prioridad.
                retained = [p for p in payload["predictions"] if p["confidence"] >= threshold]
                truth = payload["ground_truth"]
                matched = {p["matched_gt_index"] for p in retained if p["status"] == "tp"}
                row = selected.loc[payload["filename"]]
                for class_id, cls in enumerate(("smoke", "fire")):
                    tp = sum(p["class_id"] == class_id and p["status"] == "tp" for p in retained)
                    fp = sum(p["class_id"] == class_id and p["status"] == "fp" for p in retained)
                    fn = sum(g["class_id"] == class_id for g in truth) - tp
                    for field, value in (("tp", tp), ("fp", fp), ("fn", fn)):
                        assert row[f"{cls}_{field}"] == value
                        counts[cls][field] += value
                if not truth:
                    alarms += bool(retained)
                    negative_boxes += len(retained)
                for index, g in enumerate(truth):
                    box = g["xyxy"]
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    size = "small" if area < config["size_area_boundaries"][0] else "medium" if area < config["size_area_boundaries"][1] else "large"
                    bucket = size_counts[(("smoke", "fire")[g["class_id"]], size)]
                    bucket[0] += 1
                    bucket[1] += index in matched
                verified_rows += 1
            metric = model_metrics[np.isclose(model_metrics.threshold, threshold)].iloc[0]
            assert metric.negative_images_with_alarm == alarms
            assert np.isclose(metric.negative_alarm_rate, alarms / 783)
            assert metric.negative_false_boxes == negative_boxes
            counts["micro"] = {key: sum(counts[cls][key] for cls in ("smoke", "fire")) for key in ("tp", "fp", "fn")}
            for cls, values in counts.items():
                for field in ("tp", "fp", "fn"):
                    assert metric[f"{cls}_{field}"] == values[field]
                tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
                for field, numerator, denominator in (("precision", tp, tp + fp), ("recall", tp, tp + fn), ("f1", 2 * tp, 2 * tp + fp + fn)):
                    actual = metric[f"{cls}_{field}"]
                    assert np.isclose(actual, numerator / denominator) if denominator else pd.isna(actual)
            for (cls, size), (total, tp) in size_counts.items():
                row = sizes[(sizes.model_key == model_key) & np.isclose(sizes.threshold, threshold) & (sizes.class_name == cls) & (sizes.size_band == size)].iloc[0]
                assert row.gt_boxes == total and row.tp == tp and row.fn == total - tp
        # Monotonía de TP y alarmas frente al umbral, sin asumir monotonía de precisión.
        for field in ("smoke_tp", "fire_tp", "negative_images_with_alarm"):
            assert (model_metrics.sort_values("threshold")[field].diff().dropna() <= 0).all()
        historical = summary["baseline_verification"][model_key]
        assert historical["status"] == "passed"
        assert pipeline.sha256_file(ROOT / historical["reference_summary_rel"]) == historical["reference_summary_sha256"]
    for candidate in candidates.itertuples():
        table = metrics[metrics.model_key == candidate.model_key]
        if candidate.scenario.startswith("alarm_"):
            eligible = table[table.negative_images_with_alarm <= candidate.alarm_budget * table.negative_images + 1e-12]
            assert bool(candidate.feasible) == (not eligible.empty)
            if candidate.feasible:
                score = lambda r: (r.macro_recall, r.minimum_class_recall, r.micro_f1, -r.negative_images_with_alarm, r.threshold)
                winner = max(eligible.itertuples(), key=score)
                assert winner.threshold == candidate.threshold
        elif candidate.scenario == "max_f1":
            assert np.isclose(candidate.micro_f1, table.micro_f1.max())
    review = pd.read_csv(output / "review_selection.csv")
    for filename in review.filename.unique():
        row = validation.loc[filename]
        source = ROOT / "data" / "D-Fire" / row.official_split / "images" / filename
        assert pipeline.sha256_file(source) == row.sha256
        verified_source_images += 1
    # Los archivos generados no pueden cambiar silenciosamente tras completar el barrido.
    for name, checksum in summary["output_hashes"].items():
        assert pipeline.sha256_file(output / name) == checksum, name
    report = {"status": "passed", "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "threshold_points": len(metrics), "per_image_threshold_rows": verified_rows,
              "cached_matches_checked_by_independent_iou": verified_matches,
              "reviewed_original_image_hashes": verified_source_images,
              "scenario_rules": "verified", "reference_025": "exact_per_image_reconciliation",
              "method": "Independent counts from low-confidence inference matching; no sweep/matching function calls.",
              "inference_repeated": False}
    if record:
        pipeline.write_json_atomic(output / "verification.json", report)
        visual_path = output / "visual_review.csv"
        if visual_path.exists():
            visual = pd.read_csv(visual_path)
            assert not visual[["model_key", "filename"]].duplicated().any()
            assert set(zip(visual.model_key, visual.filename)) == set(zip(review.model_key, review.filename))
            assert visual.observation.notna().all() and visual.review_status.eq("inspected").all()
            summary["visual_review"] = {"status": "complete_directed_sample", "model_image_pairs": len(visual),
                                        "unique_images": visual.filename.nunique(), "note": "Not an exhaustive annotation audit."}
        summary["independent_verification"] = report
        summary["output_hashes"] = {p.relative_to(output).as_posix(): pipeline.sha256_file(p)
                                    for p in sorted(output.rglob("*")) if p.is_file()
                                    and p.name not in {"run_summary.json", "notebook05.html"}}
        pipeline.write_json_atomic(summary_path, summary)
        latest = pipeline.read_json(parent / "latest.json")
        if latest["run_id"] == summary["run_id"]:
            latest["summary_sha256"] = pipeline.sha256_file(summary_path)
            pipeline.write_json_atomic(parent / "latest.json", latest)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--record", action="store_true", help="Guarda verificación e indexa la revisión visual añadida.")
    args = parser.parse_args()
    verify(args.directory, record=args.record)
