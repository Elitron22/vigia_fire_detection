import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

import tfm_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_class_mapping_is_fixed(self):
        self.assertEqual(
            pipeline.validate_class_mapping({"0": "Smoke", "1": "FIRE"}),
            {0: "smoke", 1: "fire"},
        )
        with self.assertRaises(ValueError):
            pipeline.validate_class_mapping({0: "fire", 1: "smoke"})

    def test_registry_has_planned_models(self):
        registry = pipeline.load_model_registry(Path(__file__).parents[1])
        self.assertTrue(
            {"yolov8n", "yolov8s", "yolo26n", "yolo26s"}.issubset(
                registry["models"]
            )
        )
        self.assertIn("controlled", registry["profiles"])

    def test_runtime_yaml_is_rebased(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = {"prepared_root": root / "prepared"}
            path = pipeline.write_runtime_data_yaml(contract, root / "runtime.yaml")
            text = path.read_text(encoding="utf-8")
            self.assertIn(str((root / "prepared" / "dataset").resolve()), text)
            self.assertIn("train: images/train", text)
            self.assertIn("0: smoke", text)

    def test_rebased_manifest_ignores_historical_absolute_paths(self):
        contract = {
            "prepared_root": Path("/new/root"),
            "source_dataset_root": Path("/new/root"),
            "manifest": pd.DataFrame(
                [{
                    "split": "test",
                    "filename": "image.jpg",
                    "image_path": "/old/root/image.jpg",
                    "label_path": "/old/root/image.txt",
                }]
            ),
        }
        result = pipeline.rebased_manifest(contract).iloc[0]
        self.assertTrue(result.image_path.endswith("test/images/image.jpg"))
        self.assertTrue(result.label_path.endswith("test/labels/image.txt"))

    def test_experiment_ids_are_safe_and_distinct_format(self):
        experiment_id = pipeline.utc_experiment_id("my model", 42)
        self.assertRegex(experiment_id, r"^my-model_dfire_seed42_\d{8}T\d{6}Z$")

    def test_fast_staging_applies_repairs_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            prepared = root / "prepared"
            image_path = source / "train" / "images" / "sample.jpg"
            label_path = source / "train" / "labels" / "sample.txt"
            image_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "red").save(image_path)
            label_path.write_text("0 0.5 0.5 1.2 1.0\n1 0.5 0.5 0 0\n", encoding="utf-8")
            manifest = pd.DataFrame([{
                "split": "train", "official_split": "train",
                "filename": "sample.jpg", "image_path": str(image_path),
                "label_path": str(label_path), "jpeg_missing_eoi": False,
            }])
            prepared.mkdir()
            manifest_path = prepared / "dataset_manifest.csv"
            manifest.to_csv(manifest_path, index=False)
            pd.DataFrame([{
                "official_split": "train", "label_path": str(label_path),
                "line_number": 1, "class_id": 0,
                "clipped_x_center": 0.5, "clipped_y_center": 0.5,
                "clipped_width": 1.0, "clipped_height": 1.0,
            }]).to_csv(prepared / "annotation_repairs.csv", index=False)
            pd.DataFrame([{
                "official_split": "train", "label_path": str(label_path),
                "line_number": 2,
            }]).to_csv(prepared / "annotation_dropped_boxes.csv", index=False)
            contract = {
                "prepared_root": prepared,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "metadata": {"dataset_version": "synthetic_v1"},
                "source_dataset_root": source,
            }
            staged = pipeline.stage_prepared_dataset(
                contract, fast_base=root / "fast", workers=2
            )
            staged_label = staged / "labels" / "train" / "sample.txt"
            self.assertEqual(staged_label.read_text().strip(), "0 0.5 0.5 1 1")
            self.assertEqual(
                pipeline.stage_prepared_dataset(contract, fast_base=root / "fast"),
                staged,
            )


if __name__ == "__main__":
    unittest.main()
