from __future__ import annotations

import unittest

import numpy as np

from tools.run_operational_video_benchmark import states_to_episodes, temporal_states


class TemporalRuleTests(unittest.TestCase):
    def test_three_of_five_activates_on_third_hit(self) -> None:
        raw = [True, False, True, False, True, False]
        observed = temporal_states(raw, window_frames=5, minimum_hits=3, clear_after_misses=10)
        np.testing.assert_array_equal(observed, [False, False, False, False, True, True])

    def test_active_alarm_closes_after_consecutive_misses(self) -> None:
        raw = [True, True, False, True, False, False, False]
        observed = temporal_states(raw, window_frames=3, minimum_hits=2, clear_after_misses=3)
        np.testing.assert_array_equal(observed, [False, True, True, True, True, True, False])

    def test_isolated_hits_do_not_trigger_two_of_three(self) -> None:
        raw = [True, False, False, True, False, False]
        observed = temporal_states(raw, window_frames=3, minimum_hits=2, clear_after_misses=2)
        self.assertFalse(observed.any())

    def test_episode_boundaries_use_timestamps(self) -> None:
        states = np.asarray([False, True, True, False, False, True], dtype=bool)
        timestamps = np.arange(len(states), dtype=float) * .2
        episodes = states_to_episodes(states, timestamps, .2)
        self.assertEqual(len(episodes), 2)
        self.assertAlmostEqual(episodes[0]["start_s"], .2)
        self.assertAlmostEqual(episodes[0]["end_s"], .6)
        self.assertAlmostEqual(episodes[1]["start_s"], 1.0)
        self.assertAlmostEqual(episodes[1]["end_s"], 1.2)


if __name__ == "__main__":
    unittest.main()
