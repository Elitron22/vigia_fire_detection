import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from tools.run_resolution_comparison import exact_mcnemar, image_detection_metrics, load_config


class ResolutionComparisonTests(unittest.TestCase):
    def test_exact_mcnemar_counts_paired_changes(self):
        result = exact_mcnemar([True, False, False, True], [True, True, False, False])
        self.assertEqual(result["gains"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["discordant"], 2)
        self.assertEqual(result["p_value"], 1.0)

    def test_image_detection_recall_requires_a_matched_box(self):
        table = pd.DataFrame([
            {"smoke_gt": 0, "fire_gt": 2, "gt_count": 2, "smoke_tp": 0, "fire_tp": 1},
            {"smoke_gt": 0, "fire_gt": 1, "gt_count": 1, "smoke_tp": 0, "fire_tp": 0},
            {"smoke_gt": 0, "fire_gt": 0, "gt_count": 0, "smoke_tp": 0, "fire_tp": 0},
        ])
        result = image_detection_metrics(table)
        self.assertEqual(result["fire_positive_images"], 2)
        self.assertEqual(result["fire_images_detected"], 1)
        self.assertEqual(result["fire_image_recall"], 0.5)

    def test_config_rejects_missing_factorial_size(self):
        source = Path(__file__).parents[1] / "configs" / "resolution_comparison.yaml"
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["evaluation_sizes"] = [640, 1024]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
