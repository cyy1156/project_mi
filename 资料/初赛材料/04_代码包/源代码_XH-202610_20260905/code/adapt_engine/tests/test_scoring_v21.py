"""D8 v2.1 在线计分单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapt_engine.scoring_v21 import (  # noqa: E402
    OnlineScoreTracker,
    ScoringConfig,
    build_judgment_times,
    score_trial_from_judgments,
    tick_weight,
)


CFG = ScoringConfig()


def test_build_judgment_times_06_grid():
    times = build_judgment_times(0.6, 6.0)
    assert times == (0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0)
    assert len(times) == 10


def test_tick_weight_half_and_full():
    assert tick_weight(2.4, 2.4) == 0.5
    assert tick_weight(3.0, 2.4) == 1.0


def test_early_stop_at_score_5():
    """全对时最早 mi+4.2s 达 5 分早停。"""
    tracker = OnlineScoreTracker(1, CFG)
    preds = [1] * 7
    times = build_judgment_times(0.6, 6.0)[:7]
    stop_t = None
    for t, p in zip(times, preds):
        tick = tracker.apply_tick(t, p)
        if tick["early_stop"]:
            stop_t = t
            break
    assert stop_t == 4.2
    verdict = tracker.finalize(ended_early=True, end_reason="score_5")
    assert verdict.valid and verdict.early_stop


def test_wrong_class_abort():
    """错类累计加权 ≥5 → 无效 A。"""
    label = 1
    judgments = [{"t": t, "pred": 2} for t in build_judgment_times(0.6, 6.0)[:9]]
    verdict = score_trial_from_judgments(label, judgments, CFG, ended_early=True)
    assert not verdict.valid
    assert verdict.invalid_reason == "trial_invalid_wrong_race"


def test_full_6s_low_score_invalid():
    """满 6s 但得分 ≤3 → 无效 B。"""
    times = build_judgment_times(0.6, 6.0)
    preds = [1, 1, 1, 1, 1, 0, 2, 0, 2, 0]
    judgments = [{"t": t, "pred": p} for t, p in zip(times, preds)]
    verdict = score_trial_from_judgments(1, judgments, CFG, ended_early=False, end_reason="full_6s")
    assert not verdict.valid
    assert verdict.invalid_reason == "trial_invalid_low_score"
    assert verdict.score <= 3.0


def test_full_6s_score_4_valid():
    """满 6s 得分 4 → 有效（未满 5 亦可）。"""
    times = build_judgment_times(0.6, 6.0)
    preds = [1] * 6 + [0, 2, 0, 2]
    judgments = [{"t": t, "pred": p} for t, p in zip(times, preds)]
    verdict = score_trial_from_judgments(1, judgments, CFG, ended_early=False, end_reason="full_6s")
    assert verdict.valid
    assert verdict.score == 4.0
