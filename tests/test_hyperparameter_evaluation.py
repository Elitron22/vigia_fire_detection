import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import tfm_pipeline as pipeline
from tools.run_hyperparameter_evaluation import (
    DEFAULT_CONFIG, load_config, make_figures, select_operating_points,
    select_preliminary, training_summary,
)


class HyperparameterEvaluationTests(unittest.TestCase):
    def test_config_is_locked_and_bounded(self):
        config = load_config(DEFAULT_CONFIG)
        self.assertTrue(config["selection"]["test_locked"])
        self.assertEqual(config["evaluation"]["split"], "val")
        self.assertEqual(config["evaluation"]["candidate_count"], 2)
        self.assertEqual(len(config["active_trials"]), 4)

    def test_preliminary_selection_uses_declared_ordering(self):
        table = pd.DataFrame([
            {"trial_id": "a", "best_map5095_50": .45, "last5_mean_map5095": .44, "recall_at_best_50": .70},
            {"trial_id": "b", "best_map5095_50": .47, "last5_mean_map5095": .45, "recall_at_best_50": .71},
            {"trial_id": "c", "best_map5095_50": .46, "last5_mean_map5095": .455, "recall_at_best_50": .72},
        ])
        selected = select_preliminary(table, 2)
        self.assertEqual(selected.trial_id.tolist(), ["b", "c", "a"])
        self.assertEqual(selected[selected.selected_for_operational_evaluation].trial_id.tolist(), ["b", "c"])

    def test_training_summary_uses_first_50_baseline_epochs(self):
        config = load_config(DEFAULT_CONFIG)
        baseline = pipeline.resolve_experiment(
            experiment_id=config["baseline_experiment_id"], root=DEFAULT_CONFIG.parents[1]
        )
        keys = ["baseline", *config["active_trials"]]
        summary, curves = training_summary({key: baseline for key in keys}, config)
        self.assertEqual(len(summary), 5)
        self.assertTrue(summary.epochs_compared.eq(50).all())
        self.assertLessEqual(summary.best_epoch_50.max(), 50)
        self.assertEqual(curves.groupby("trial_id").size().tolist(), [50] * 5)

    def test_operating_selection_respects_alarm_budget(self):
        rows = []
        for profile in ("a", "b"):
            rows.extend([
                {"profile": profile, "smoke_threshold": .1, "fire_threshold": .1,
                 "macro_recall": .9, "minimum_class_recall": .85, "micro_f1": .65,
                 "micro_precision": .55, "negative_images": 100,
                 "negative_images_with_alarm": 3},
                {"profile": profile, "smoke_threshold": .2, "fire_threshold": .2,
                 "macro_recall": .8, "minimum_class_recall": .78, "micro_f1": .75,
                 "micro_precision": .72, "negative_images": 100,
                 "negative_images_with_alarm": 2},
                {"profile": profile, "smoke_threshold": .3, "fire_threshold": .3,
                 "macro_recall": .7, "minimum_class_recall": .68, "micro_f1": .80,
                 "micro_precision": .82, "negative_images": 100,
                 "negative_images_with_alarm": 1},
            ])
        selected = select_operating_points(pd.DataFrame(rows), .02)
        self.assertEqual(len(selected), 4)
        self.assertTrue(selected.negative_images_with_alarm.le(2).all())
        sensitivity = selected[selected.scenario == "sensitivity"]
        balanced = selected[selected.scenario == "balanced"]
        self.assertTrue(sensitivity.smoke_threshold.eq(.2).all())
        self.assertTrue(balanced.smoke_threshold.eq(.3).all())

    def test_figures_render_from_complete_synthetic_tables(self):
        trials = ["baseline", "hp01_lr5e4"]
        training = pd.DataFrame({
            "trial_id": trials, "best_map5095_50": [.45, .47],
        })
        curves = pd.concat([
            pd.DataFrame({"trial_id": trial, "epoch": np.arange(1, 51),
                          "metrics/mAP50-95(B)": np.linspace(.1, value, 50)})
            for trial, value in zip(trials, [.45, .47])
        ], ignore_index=True)
        recommendations = pd.DataFrame([
            {"trial_id": trial, "scenario": scenario, "smoke_recall": .82,
             "fire_recall": .76, "micro_precision": .70, "micro_f1": .73}
            for trial in trials for scenario in ("sensitivity", "balanced")
        ])
        grid = pd.DataFrame([
            {"profile": trial, "smoke_threshold": smoke, "fire_threshold": fire,
             "micro_f1": .70 + smoke / 10 + fire / 10}
            for trial in trials for smoke in (.1, .2) for fire in (.1, .2)
        ])
        with tempfile.TemporaryDirectory() as directory:
            paths = make_figures(training, curves, recommendations, grid, Path(directory))
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
