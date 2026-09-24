"""Verificación independiente de los notebooks y artefactos reorganizados."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tfm_pipeline as pipeline


def main() -> None:
    contract = pipeline.validate_prepared_dataset(check_all_images=True)
    manifest = contract["manifest"]
    expected_splits = {"train": 15_500, "val": 1_721, "test": 4_306}
    assert manifest["split"].value_counts().to_dict() == expected_splits
    assert int(manifest["smoke_boxes"].sum()) == 11_854
    assert int(manifest["fire_boxes"].sum()) == 14_685

    staged = (
        pipeline.default_fast_data_root(ROOT)
        / contract["metadata"]["dataset_version"]
        / "dataset"
    )
    marker = staged.parent / "stage_complete.json"
    assert marker.exists()
    assert pipeline.read_json(marker)["manifest_sha256"] == pipeline.sha256_file(
        contract["manifest_path"]
    )

    image_count = sum(1 for path in (staged / "images").rglob("*") if path.is_file())
    label_files = [path for path in (staged / "labels").rglob("*.txt") if path.is_file()]
    assert image_count == len(manifest) == 21_527
    assert len(label_files) == len(manifest)

    boxes = [0, 0]
    for label in label_files:
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            assert len(values) == 5, (label, line_number)
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
            assert class_id in (0, 1)
            x, y, width, height = coordinates
            assert 0 <= x <= 1 and 0 <= y <= 1 and width > 0 and height > 0
            assert x - width / 2 >= -1e-9 and y - height / 2 >= -1e-9
            assert x + width / 2 <= 1 + 1e-9 and y + height / 2 <= 1 + 1e-9
            boxes[class_id] += 1
    assert boxes == [int(manifest.smoke_boxes.sum()), int(manifest.fire_boxes.sum())]

    missing_eoi = manifest[manifest["jpeg_missing_eoi"].astype(bool)]
    assert len(missing_eoi) == 37
    for row in missing_eoi.itertuples():
        image = staged / "images" / row.split / row.filename
        assert image.read_bytes()[-2:] == b"\xff\xd9"

    notebook_status = {}
    transformer = TransformerManager()
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        errors = [
            output
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, (path, errors)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(transformer.transform_cell(cell.source))
        parameter_cells = [
            cell for cell in notebook.cells if "parameters" in cell.metadata.get("tags", [])
        ]
        if parameter_cells:
            source = parameter_cells[0].source
            assert "RUN_TRAINING = True" not in source
            assert "RUN_STANDARD_EVALUATION = True" not in source
            assert "RUN_ERROR_ANALYSIS = True" not in source
            assert "RUN_PREPARATION = True" not in source
        notebook_status[path.name] = {
            "cells": len(notebook.cells),
            "executed_code_cells": sum(
                cell.cell_type == "code" and cell.get("execution_count") is not None
                for cell in notebook.cells
            ),
        }

    result = {
        "status": "passed",
        "prepared_images": len(manifest),
        "splits": expected_splits,
        "staged_images": image_count,
        "staged_label_files": len(label_files),
        "staged_boxes": {"smoke": boxes[0], "fire": boxes[1]},
        "jpeg_eoi_repaired_in_staging": len(missing_eoi),
        "original_dataset_modified": False,
        "notebooks": notebook_status,
    }
    destination = contract["prepared_root"] / "pipeline_verification.json"
    pipeline.write_json_atomic(destination, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
