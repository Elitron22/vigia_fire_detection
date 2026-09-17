"""Comprueba FP32 actual frente a 8 predicciones de la ejecución completada."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tfm_evaluation import iter_bounded_predictions, sha256_file

directory = Path(sys.argv[1])
summary = json.loads((directory / "test_error_summary.json").read_text())
table = pd.read_csv(directory / "test_error_analysis.csv")
selected = pd.concat([table.head(4), table.loc[table.is_negative & (table.pred_count > 0)].head(4)])
manifest = pd.read_csv(summary["manifest_path"])
selected_manifest = manifest.loc[manifest.image_path.isin(selected.image_path)]
expected = {}
with (directory / "test_predictions.jsonl").open() as handle:
    for line in handle:
        item = json.loads(line)
        if item["image_path"] in set(selected.image_path):
            expected[item["image_path"]] = item["predictions"]
model = YOLO(summary["model_path"])
checked = 0
for record, actual in iter_bounded_predictions(model, selected_manifest.to_dict("records"), chunk_size=4):
    saved = expected[record["image_path"]]
    assert len(saved) == len(actual), record["image_path"]
    actual = sorted(actual, key=lambda p: -p["confidence"])
    for left, right in zip(saved, actual):
        assert left["class_id"] == right["class_id"]
        assert abs(left["confidence"] - right["confidence"]) < 1e-4
        assert np.allclose(left["xyxy"], right["xyxy"], atol=1e-5)
    checked += 1
report = {"status": "passed", "images_checked": checked, "precision": "FP32",
          "note": "quantize=fp32 reemplaza half=False sin cambiar el protocolo; predicciones contrastadas con la ejecución completa.",
          "current_code_sha256": sha256_file(Path(__file__).resolve().parents[1] / "tfm_evaluation.py")}
(directory / "inference_smoke_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
