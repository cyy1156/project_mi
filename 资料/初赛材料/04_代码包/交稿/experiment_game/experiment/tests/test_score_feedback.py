"""score_feedback：第 1 次命中即可见伸手。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiment_game.experiment.score_feedback import (
    arm_progress_from_score,
    enrich_stage_data,
    map_score_to_arm,
)


class TestMapScoreToArm(unittest.TestCase):
    def test_zero_no_reach(self) -> None:
        self.assertEqual(map_score_to_arm(0), (0, False))

    def test_first_hit_visible(self) -> None:
        level, grasp = map_score_to_arm(1.0)
        self.assertEqual(level, 1)
        self.assertFalse(grasp)
        self.assertGreater(level / 4.0, 0.0)

    def test_mid_scores(self) -> None:
        self.assertEqual(map_score_to_arm(2)[0], 2)
        self.assertEqual(map_score_to_arm(3)[0], 3)
        self.assertEqual(map_score_to_arm(4)[0], 3)  # n_levels-1

    def test_grasp_at_five(self) -> None:
        self.assertEqual(map_score_to_arm(5), (4, True))
        self.assertEqual(map_score_to_arm(6), (4, True))

    def test_progress_first_hit(self) -> None:
        p = arm_progress_from_score(1.0)
        self.assertAlmostEqual(p, 0.2)
        self.assertAlmostEqual(arm_progress_from_score(5.0), 1.0)


class TestEnrichStageData(unittest.TestCase):
    def test_judge_injects_progress(self) -> None:
        ctx = SimpleNamespace(trial_id=1, label=1)
        peaks: dict = {}
        out = enrich_stage_data("judge", ctx, {"score": 1.0}, peak_by_trial=peaks)
        assert out is not None
        self.assertEqual(out["arm_level"], 1)
        self.assertFalse(out["cup_grasp"])
        self.assertAlmostEqual(out["arm_progress"], 0.2)
        self.assertEqual(peaks[1], 1)

    def test_rest_no_arm(self) -> None:
        ctx = SimpleNamespace(trial_id=2, label=0)
        out = enrich_stage_data("judge", ctx, {"score": 3.0})
        assert out is not None
        self.assertEqual(out["arm_level"], 0)
        self.assertFalse(out["cup_grasp"])
        self.assertEqual(out["arm_progress"], 0.0)

    def test_monotonic_peak(self) -> None:
        ctx = SimpleNamespace(trial_id=3, label=2)
        peaks = {3: 2.0}  # peak score
        out = enrich_stage_data("judge", ctx, {"score": 1.0}, peak_by_trial=peaks)
        assert out is not None
        self.assertEqual(out["arm_level"], 2)
        self.assertAlmostEqual(peaks[3], 2.0)

    def test_signal_bad_progress_does_not_shrink(self) -> None:
        ctx = SimpleNamespace(trial_id=4, label=1)
        peaks: dict = {}
        enrich_stage_data("judge", ctx, {"score": 4.0}, peak_by_trial=peaks)
        out = enrich_stage_data(
            "judge", ctx, {"score": 4.0, "signal_bad": True}, peak_by_trial=peaks
        )
        assert out is not None
        self.assertEqual(out["arm_level"], 3)
        self.assertAlmostEqual(out["arm_progress"], 0.8)
        self.assertFalse(out["cup_grasp"])


if __name__ == "__main__":
    unittest.main()
