"""Pruebas sintéticas: no usan D-Fire ni GPU ni descargan pesos."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from tfm_evaluation import (box_iou, image_error_record, iter_bounded_predictions,
                            match_detections, read_ground_truth,
                            run_error_analysis, split_artifact_paths,
                            summarize_errors)


def gt(cls=0, box=(.1, .1, .9, .9)):
    return {"class_id": cls, "xyxy": list(box)}


def pred(cls=0, confidence=.9, box=(.1, .1, .9, .9)):
    return {**gt(cls, box), "confidence": confidence}


def record(name, category="background"):
    return {"image_path": f"/images/{name}.jpg", "label_path": f"/labels/{name}.txt",
            "filename": name + ".jpg", "category": category, "height": 10, "width": 10}


class FakeModel:
    names = {0: "smoke", 1: "fire"}

    def __init__(self, fault=None):
        self.calls = []
        self.fault = fault
        self.predictor = SimpleNamespace(dataset=None, results=None, batch=None)

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        paths = kwargs["source"]
        self.predictor.dataset = paths
        if self.fault == "missing":
            paths = paths[:-1]
        if self.fault == "extra":
            paths = paths + paths[:1]
        for path in paths:
            if self.fault == "interrupt" and len(self.calls) == 2:
                raise RuntimeError("Interrupción simulada")
            yield SimpleNamespace(path="/wrong.jpg" if self.fault == "path" else path,
                                  orig_shape=(10, 10), boxes=None)


class EvaluationTests(unittest.TestCase):
    def test_iou_boundary(self):
        self.assertEqual(box_iou([0, 0, 1, 1], [0, 0, .5, 1]), .5)
        detections, missed = match_detections([gt(box=(0, 0, 1, 1))], [pred(box=(0, 0, .5, 1))])
        self.assertEqual(detections[0]["status"], "tp")
        self.assertEqual(missed, [])

    def test_highest_confidence_and_one_to_one(self):
        detections, missed = match_detections([gt()], [pred(confidence=.3), pred(confidence=.9)])
        self.assertEqual([d["status"] for d in detections], ["tp", "fp"])
        self.assertEqual(detections[0]["confidence"], .9)
        self.assertEqual(missed, [])

    def test_wrong_class_is_fp_and_fn(self):
        detections, missed = match_detections([gt(0)], [pred(1)])
        self.assertEqual(detections[0]["status"], "fp")
        self.assertEqual(missed, [0])

    def test_background_alarm_is_image_not_boxes(self):
        rows = []
        samples = [("neg1", [], [pred(), pred(1), pred(1)]), ("neg2", [], []),
                   ("pos", [gt()], [pred(), pred(1)])]
        for name, truth, predictions in samples:
            detections, missed = match_detections(truth, predictions)
            rows.append(image_error_record(record(name), truth, detections, missed))
        summary = summarize_errors(pd.DataFrame(rows))
        self.assertEqual(summary["negative_images"], 2)
        self.assertEqual(summary["negative_image_alarms"][0]["negative_images_with_alarm"], 1)
        self.assertEqual(summary["negative_image_alarms"][0]["negative_image_false_alarm_rate"], .5)
        self.assertEqual(summary["negative_false_positive_boxes"]["all"], 3)
        self.assertEqual(summary["box_metrics"][-1]["fp"], 4)

    def test_no_negatives_is_undefined_not_zero(self):
        detections, missed = match_detections([gt()], [])
        summary = summarize_errors(pd.DataFrame([image_error_record(record("p"), [gt()], detections, missed)]))
        self.assertIsNone(summary["negative_image_alarms"][0]["negative_image_false_alarm_rate"])
        self.assertIsNone(summary["box_metrics"][0]["precision"])
        self.assertEqual(summary["box_metrics"][0]["recall"], 0.)

    def test_corrupt_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            self.assertEqual(read_ground_truth(path, 0), [])
            with self.assertRaises(FileNotFoundError):
                read_ground_truth(path, 1)
            path.write_text("0 0.5 0.5 1 1\n")
            self.assertEqual(len(read_ground_truth(path, 1)), 1)
            for content in ("2 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .5 .5 0 .2", "0 .1 .5 .9 .2", "0 .5 .5"):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_ground_truth(path)

    def test_bounded_batches_and_cleanup(self):
        model = FakeModel()
        output = list(iter_bounded_predictions(model, [record(str(i)) for i in range(5)], chunk_size=2))
        self.assertEqual(len(output), 5)
        self.assertEqual([len(c["source"]) for c in model.calls], [2, 2, 1])
        self.assertTrue(all(c["rect"] is False and c["stream"] is True for c in model.calls))
        self.assertIsNone(model.predictor.dataset)

    def test_end_to_end_model_does_not_receive_nms_iou(self):
        model = FakeModel()
        model.model = SimpleNamespace(model=[SimpleNamespace(end2end=True)])
        output = list(iter_bounded_predictions(model, [record("a")], chunk_size=1, nms_iou=.7))
        self.assertEqual(len(output), 1)
        self.assertNotIn("iou", model.calls[0])

    def test_split_specific_artifacts_do_not_mix_validation_and_test(self):
        val_paths = split_artifact_paths("output", "val")
        test_paths = split_artifact_paths("output", "test")
        self.assertEqual(val_paths["summary"].name, "val_error_summary.json")
        self.assertEqual(test_paths["summary"].name, "test_error_summary.json")
        self.assertNotEqual(val_paths["predictions"], test_paths["predictions"])
        with self.assertRaises(ValueError):
            split_artifact_paths("output", "invalid")

    def test_short_extra_or_misaligned_stream_is_blocked(self):
        for fault in ("missing", "extra", "path"):
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                list(iter_bounded_predictions(FakeModel(fault), [record("a"), record("b")], chunk_size=2))

    def test_invalid_chunk(self):
        with self.assertRaises(ValueError):
            list(iter_bounded_predictions(FakeModel(), [record("a")], chunk_size=0))

    def test_inverted_mapping_is_blocked(self):
        model = FakeModel()
        model.names = {0: "fire", 1: "smoke"}
        with self.assertRaises(ValueError):
            run_error_analysis(model, pd.DataFrame(), "not-created")

    def test_interruption_retains_partial_but_no_final_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for i in range(3):
                image = root / f"image{i}.jpg"
                label = root / f"image{i}.txt"
                image.write_bytes(b"synthetic-image-no-decode-in-fake-model")
                label.write_text("")
                records.append({**record(str(i)), "image_path": str(image), "label_path": str(label),
                                "split": "test", "box_count": 0, "smoke_boxes": 0, "fire_boxes": 0})
            with self.assertRaisesRegex(RuntimeError, "Interrupción"):
                run_error_analysis(FakeModel("interrupt"), pd.DataFrame(records), root / "runs", chunk_size=2)
            output = next((root / "runs").iterdir())
            state = json.loads((output / "run_state.json").read_text())
            self.assertEqual(state["status"], "incomplete")
            self.assertEqual(state["completed_images"], 2)
            self.assertFalse((output / "test_error_summary.json").exists())
            self.assertEqual(len(pd.read_csv(output / "test_error_analysis.csv")), 2)


if __name__ == "__main__":
    unittest.main()
