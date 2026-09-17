"""Verificación independiente de los CSV, emparejamientos y cobertura del test."""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def verify(directory, project):
    directory = Path(directory)
    summary = json.loads((directory / "test_error_summary.json").read_text())
    assert summary["status"] == "complete"
    assert digest(summary["model_path"]) == summary["model_sha256"]
    assert digest(summary["manifest_path"]) == summary["manifest_sha256"]
    assert digest(directory / "evaluation_code.py") == summary["evaluation_code_sha256"]
    assert digest(directory / "input_inventory.csv") == summary["input_inventory_sha256"]
    manifest = pd.read_csv(summary["manifest_path"])
    manifest = manifest.loc[manifest.split == "test"]
    images = pd.read_csv(directory / "test_error_analysis.csv")
    detections = pd.read_csv(directory / "test_detection_details.csv")
    truth = pd.read_csv(directory / "test_ground_truth_details.csv")
    assert len(images) == len(manifest) == 4306
    assert not images.image_path.duplicated().any()
    assert set(images.image_path) == set(manifest.image_path)
    assert len(truth) == int(manifest.box_count.sum()) == 5189
    assert not truth[["image_path", "gt_index"]].duplicated().any()
    assert set(detections.status) <= {"tp", "fp"}
    assert set(truth.status) <= {"tp", "fn"}
    assert detections.confidence.between(.25, 1).all()
    for frame in (truth, detections):
        assert frame.class_id.isin([0, 1]).all()
        assert frame[["x1", "y1", "x2", "y2"]].ge(-1e-6).all().all()
        assert frame[["x1", "y1", "x2", "y2"]].le(1 + 1e-6).all().all()
    for class_id, name in enumerate(("smoke", "fire")):
        pred_class = detections.loc[detections.class_id == class_id]
        gt_class = truth.loc[truth.class_id == class_id]
        for status in ("tp", "fp", "fn"):
            frame = gt_class if status == "fn" else pred_class
            counts = frame.loc[frame.status == status].groupby("image_path").size()
            actual = images.image_path.map(counts).fillna(0).astype(int)
            assert (actual == images[f"{name}_{status}"]).all()
    matched = detections.loc[detections.status == "tp"]
    assert not matched[["image_path", "matched_gt_index"]].duplicated().any()
    pairs = matched.merge(truth, left_on=["image_path", "matched_gt_index"],
                          right_on=["image_path", "gt_index"], suffixes=("_pred", "_gt"), validate="one_to_one")
    assert len(pairs) == len(matched) == int((truth.status == "tp").sum())
    assert (pairs.class_id_pred == pairs.class_id_gt).all()
    a = pairs[["x1_pred", "y1_pred", "x2_pred", "y2_pred"]].to_numpy()
    b = pairs[["x1_gt", "y1_gt", "x2_gt", "y2_gt"]].to_numpy()
    intersection = np.maximum(np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]), 0).prod(axis=1)
    iou = intersection / ((a[:, 2:] - a[:, :2]).prod(axis=1) + (b[:, 2:] - b[:, :2]).prod(axis=1) - intersection)
    assert (iou >= .5 - 1e-7).all()
    assert np.allclose(iou, pairs.match_iou, atol=1e-7)
    negative_paths = set(manifest.loc[manifest.box_count == 0, "image_path"])
    negative_detections = detections.loc[detections.image_path.isin(negative_paths)]
    alarm_count = negative_detections.image_path.nunique()
    assert len(negative_paths) == 2005
    assert alarm_count == summary["negative_image_alarms"][0]["negative_images_with_alarm"]
    assert len(negative_detections) == summary["negative_false_positive_boxes"]["all"]
    assert (negative_detections.status == "fp").all()
    by_class = {}
    for cls, name in enumerate(("smoke", "fire")):
        selected = negative_detections.loc[negative_detections.class_id == cls]
        by_class[name] = set(selected.image_path)
        row = next(r for r in summary["negative_image_alarms"] if r["scope"] == name)
        assert len(by_class[name]) == row["negative_images_with_alarm"]
    assert len(by_class["smoke"] & by_class["fire"]) == summary["negative_image_alarms"][3]["negative_images_with_alarm"]
    payload_rows = 0
    payload_predictions = 0
    with (directory / "test_predictions.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            payload_rows += 1
            payload_predictions += len(item["predictions"])
    assert payload_rows == len(images) and payload_predictions == len(detections)
    checked_cells = {}
    for filename in ("01_DFire_YOLOv8s_baseline.ipynb", "02_DFire_evaluacion_operativa.ipynb"):
        notebook = nbformat.read(Path(project) / filename, as_version=4)
        nbformat.validate(notebook)
        count = 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(TransformerManager().transform_cell(cell.source), f"{filename}:{index}", "exec")
                count += 1
                if filename.startswith("02_"):
                    assert cell.execution_count is not None
                    assert all(o.output_type != "error" for o in cell.outputs)
        checked_cells[filename] = count
    report = {"status": "passed", "images": len(images), "ground_truth_boxes": len(truth),
              "predicted_boxes": len(detections), "matched_boxes_verified_by_independent_iou": len(pairs),
              "negative_images": len(negative_paths), "negative_images_with_alarm": alarm_count,
              "negative_false_positive_boxes": len(negative_detections),
              "positive_image_false_positive_boxes": int(((detections.status == "fp") & ~detections.image_path.isin(negative_paths)).sum()),
              "checkpoint_manifest_and_code_hashes": "unchanged",
              "notebook_code_cells_compiled": checked_cells,
              "notebook_02_execution": "top_to_bottom_read_saved_results_passed",
              "notebook_01_execution": "syntax_checked_only; full_audit_and_training_not_repeated"}
    (directory / "independent_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verify(args.directory, args.project)
