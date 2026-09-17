"""Casos de frontera del barrido; sin GPU, D-Fire ni descargas."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import yaml

from tfm_thresholds import (
    choose_scenarios, evaluate_threshold, fp_reason, load_config, load_predictions, size_band,
)


def gt(cls=0, box=(.1, .1, .5, .5)):
    return {"class_id": cls, "xyxy": list(box)}


def pred(cls=0, confidence=.9, box=(.1, .1, .5, .5)):
    return {**gt(cls, box), "confidence": confidence, "status": "bogus_cached_status"}


def fixtures():
    samples = [
        {"filename": "positive.jpg", "ground_truth": [gt(0), gt(1, (.6, .6, .9, .9))],
         "predictions": [pred(0, .25), pred(0, .10), pred(1, .20, (.6, .6, .9, .9))]},
        {"filename": "negative_a.jpg", "ground_truth": [], "predictions": [pred(0, .3), pred(1, .4)]},
        {"filename": "negative_b.jpg", "ground_truth": [], "predictions": []},
    ]
    rows = []
    for item in samples:
        rows.append({"filename": item["filename"], "image_path": "/images/" + item["filename"],
                     "label_path": "/labels/" + item["filename"], "category": "sample", "split": "val",
                     "box_count": len(item["ground_truth"]),
                     "smoke_boxes": sum(g["class_id"] == 0 for g in item["ground_truth"]),
                     "fire_boxes": sum(g["class_id"] == 1 for g in item["ground_truth"])})
    return samples, pd.DataFrame(rows)


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "configs" / "threshold_sweep.yaml")

    def test_filter_boundary_rematching_and_negative_image_denominator(self):
        samples, manifest = fixtures()
        result, images, sizes, truth, detections = evaluate_threshold(
            samples, {r["filename"]: r for r in manifest.to_dict("records")}, .25,
            self.config, "yolov8s", details=True,
        )
        self.assertEqual((result["micro_tp"], result["micro_fp"], result["micro_fn"]), (1, 2, 1))
        self.assertEqual(result["negative_alarm_rate"], .5)
        self.assertEqual(result["negative_false_boxes"], 2)
        self.assertAlmostEqual(result["micro_precision"], 1 / 3)
        self.assertEqual(result["micro_recall"], .5)
        self.assertEqual(result["micro_f1"], .4)
        self.assertEqual(truth.loc[truth.class_name == "fire", "fn_diagnostic"].iloc[0], "recovered_at_floor")
        self.assertEqual(sizes.gt_boxes.sum(), 2)
        self.assertEqual(len(images), 3)

    def test_floor_duplicate_counts_and_undefined_precision(self):
        samples, manifest = fixtures()
        records = {r["filename"]: r for r in manifest.to_dict("records")}
        low, *_ = evaluate_threshold(samples, records, .01, self.config, "yolov8s")
        high, *_ = evaluate_threshold(samples, records, .99, self.config, "yolov8s")
        self.assertEqual((low["micro_tp"], low["micro_fp"], low["micro_fn"]), (2, 3, 0))
        self.assertIsNone(high["micro_precision"])
        self.assertEqual(high["micro_recall"], 0)
        self.assertEqual(high["micro_f1"], 0)

    def test_geometric_diagnostics(self):
        self.assertEqual(fp_reason(pred(0), [gt(0)], .5), "duplicate")
        self.assertEqual(fp_reason(pred(0), [gt(1)], .5), "wrong_class")
        self.assertEqual(fp_reason(pred(0, box=(.1, .1, .25, .5)), [gt(0)], .5), "localization")
        self.assertEqual(fp_reason(pred(0), [], .5), "no_overlap")

    def test_area_boundaries(self):
        self.assertEqual(size_band(gt(box=(0, 0, .1, .1))), "medium")
        self.assertEqual(size_band(gt(box=(0, 0, .1, 1))), "large")
        self.assertEqual(size_band(gt(box=(0, 0, .01, .01))), "small")

    def test_scenario_uses_unrounded_counts_and_explicit_tiebreak(self):
        table = pd.DataFrame([
            {"model_key": "yolov8s", "threshold": t, "negative_images": 783,
             "negative_images_with_alarm": alarms, "negative_alarm_rate": alarms / 783,
             "macro_recall": recall, "minimum_class_recall": recall, "micro_f1": .7}
            for t, alarms, recall in ((.20, 8, .9), (.25, 7, .8), (.30, 7, .8))
        ])
        selected = choose_scenarios(table, self.config)
        one_percent = selected[selected.scenario == "alarm_01pct"].iloc[0]
        self.assertEqual(one_percent.threshold, .30)
        self.assertEqual(one_percent.negative_images_with_alarm, 7)
        self.assertEqual(selected[selected.scenario == "alarm_02pct"].iloc[0].threshold, .20)

    def test_infeasible_is_not_silently_relaxed(self):
        table = pd.DataFrame([{"model_key": "yolov8s", "threshold": .25, "negative_images": 100,
                               "negative_images_with_alarm": 10, "negative_alarm_rate": .10,
                               "macro_recall": .9, "minimum_class_recall": .8, "micro_f1": .8}])
        rows = choose_scenarios(table, self.config)
        self.assertFalse(rows[rows.scenario == "alarm_01pct"].iloc[0].feasible)

    def test_incomplete_duplicate_wrong_partition_and_corrupt_cache_rejected(self):
        samples, manifest = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            def save(items):
                path.write_text("\n".join(json.dumps(x) for x in items), encoding="utf-8")
            save(samples)
            self.assertEqual(len(load_predictions(path, manifest, .01)), 3)
            save(samples[:2])
            with self.assertRaisesRegex(ValueError, "Cobertura incompleta"):
                load_predictions(path, manifest, .01)
            save(samples + samples[:1])
            with self.assertRaisesRegex(ValueError, "repetida"):
                load_predictions(path, manifest, .01)
            save(samples)
            with self.assertRaisesRegex(ValueError, "validación"):
                load_predictions(path, manifest.assign(split="test"), .01)
            altered = copy.deepcopy(samples)
            altered[0]["predictions"][0]["confidence"] = .001
            save(altered)
            with self.assertRaisesRegex(ValueError, "Confianza"):
                load_predictions(path, manifest, .01)

    def test_config_blocks_test_and_thresholds_below_cache_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            for field, value in (("split", "test"), ("prediction_confidence", .25), ("thresholds", [.25, .1])):
                config = {**self.config, field: value}
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_zero_area_prediction_is_preserved_as_false_positive(self):
        samples, manifest = fixtures()
        samples[1]["predictions"].append(pred(0, .30, (1, 0, 1, .5)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in samples), encoding="utf-8")
            loaded = load_predictions(path, manifest, .01)
            result, *_ = evaluate_threshold(loaded, {r["filename"]: r for r in manifest.to_dict("records")},
                                            .25, self.config, "yolov8s")
            self.assertEqual(result["micro_fp"], 3)
            self.assertEqual(result["zero_area_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
