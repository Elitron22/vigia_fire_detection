import unittest
from pathlib import Path

import nbformat
import yaml

from tools.build_hyperparameter_search_notebook import ROOT, hyperparameter_notebook


class HyperparameterSearchNotebookTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            (ROOT / "configs" / "yolo26s_hyperparameter_search.yaml").read_text(encoding="utf-8")
        )

    def test_search_is_bounded_and_keeps_test_locked(self):
        self.assertEqual(self.config["model_key"], "yolo26s")
        self.assertTrue(self.config["selection"]["test_locked"])
        active = self.config["active_trials"]
        self.assertEqual(self.config["schema_version"], 2)
        self.assertEqual(len(active), 4)
        self.assertEqual(self.config["screening_epochs"], 50)
        self.assertEqual(self.config["final_parameters"]["epochs"], 100)
        self.assertLessEqual(
            len(active) * self.config["expected_hours_per_screening_run"],
            self.config["night_budget_hours"],
        )

    def test_active_trials_are_distinct_interpretable_changes(self):
        trials = self.config["trials"]
        active = self.config["active_trials"]
        self.assertEqual(trials[active[0]]["overrides"], {"lr0": 0.0005})
        self.assertEqual(trials[active[1]]["overrides"], {"weight_decay": 0.001})
        self.assertIn("mosaic", trials[active[2]]["overrides"])
        self.assertIn("lr0", trials[active[3]]["overrides"])
        self.assertIn("mosaic", trials[active[3]]["overrides"])
        ids = [trials[key]["experiment_id"] for key in trials]
        self.assertEqual(len(ids), len(set(ids)))

    def test_notebook_is_valid_and_safe_by_default(self):
        value = hyperparameter_notebook()
        nbformat.validate(value)
        sources = "\n".join(cell.source for cell in value.cells)
        self.assertIn("RUN_TRAINING = False", sources)
        self.assertIn("RUN_EVALUATION = False", sources)
        self.assertIn("RUN_FINAL_TRAINING = False", sources)
        self.assertIn("FINAL_TRIAL_ID = None", sources)
        self.assertIn("screening_horizon", sources)
        self.assertNotIn('int(best["epoch"]) + 1', sources)
        self.assertIn("test_locked", sources)
        self.assertNotIn("split=\"test\"", sources)
        self.assertNotIn("split='test'", sources)


if __name__ == "__main__":
    unittest.main()
