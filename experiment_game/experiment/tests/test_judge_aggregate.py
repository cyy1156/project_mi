"""judge_aggregate 单测。"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from experiment_game.experiment.judge_aggregate import primary_judge_from_judgments


def test_majority_vote_wins():
    js = [
        {"pred": 1, "gated_pred": 1, "t_rel": 0.6, "p_three": [0.1, 0.7, 0.2]},
        {"pred": 1, "gated_pred": 1, "t_rel": 1.2, "p_three": [0.1, 0.6, 0.3]},
        {"pred": 2, "gated_pred": 2, "t_rel": 1.8, "p_three": [0.1, 0.2, 0.7]},
    ]
    pj = primary_judge_from_judgments(js, mode="majority", primary_s=4.0)
    assert pj is not None
    assert pj["pred"] == 1
    # F5 冻结（2026-08-29）：majority 实现含因果平滑，rule 标识为 causal_smooth_majority
    assert pj.get("rule") == "causal_smooth_majority"


def test_majority_tie_break_by_p_three():
    js = [
        {"pred": 1, "gated_pred": 1, "t_rel": 0.6, "p_three": [0.1, 0.4, 0.5]},
        {"pred": 2, "gated_pred": 2, "t_rel": 1.2, "p_three": [0.1, 0.2, 0.8]},
    ]
    pj = primary_judge_from_judgments(js, mode="majority", primary_s=4.0)
    assert pj is not None
    assert pj["pred"] == 2
