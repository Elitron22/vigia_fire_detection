from __future__ import annotations

import unittest

import pandas as pd

from tools.run_final_validation_selection import rank_candidates


CONFIG = {
    "selection_policy": {
        "maximum_negative_alarm_rate": 0.01,
        "primary_metric": "macro_recall",
        "tie_breakers": ["minimum_class_recall", "micro_f1", "micro_precision"],
    }
}


class FinalValidationSelectionTests(unittest.TestCase):
    def test_ranking_prioritizes_recall_inside_alarm_budget(self):
        table = pd.DataFrame([
            {"candidate": "a", "negative_alarm_rate": 7 / 783, "macro_recall": .79,
             "minimum_class_recall": .78, "micro_f1": .73, "micro_precision": .69},
            {"candidate": "b", "negative_alarm_rate": 7 / 783, "macro_recall": .77,
             "minimum_class_recall": .75, "micro_f1": .76, "micro_precision": .83},
        ])
        ranked = rank_candidates(table, CONFIG)
        self.assertEqual(ranked.iloc[0].candidate, "a")
        self.assertTrue(bool(ranked.iloc[0].selected))

    def test_infeasible_candidate_cannot_win(self):
        table = pd.DataFrame([
            {"candidate": "infeasible", "negative_alarm_rate": 8 / 783, "macro_recall": .95,
             "minimum_class_recall": .94, "micro_f1": .94, "micro_precision": .94},
            {"candidate": "feasible", "negative_alarm_rate": 7 / 783, "macro_recall": .70,
             "minimum_class_recall": .68, "micro_f1": .69, "micro_precision": .69},
        ])
        ranked = rank_candidates(table, CONFIG)
        self.assertEqual(ranked.iloc[0].candidate, "feasible")

    def test_selection_fails_when_no_candidate_meets_budget(self):
        table = pd.DataFrame([
            {"candidate": "a", "negative_alarm_rate": 8 / 783, "macro_recall": .80,
             "minimum_class_recall": .75, "micro_f1": .77, "micro_precision": .75},
        ])
        with self.assertRaises(ValueError):
            rank_candidates(table, CONFIG)


if __name__ == "__main__":
    unittest.main()
