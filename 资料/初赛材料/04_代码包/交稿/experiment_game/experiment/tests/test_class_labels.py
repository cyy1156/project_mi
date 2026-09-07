"""class_labels / judge 落盘字段单测。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiment_game.experiment.class_labels import (  # noqa: E402
    attach_judge_names,
    label_name,
    normalize_p_three,
)


def test_label_name():
    assert label_name(0) == "Rest"
    assert label_name(1) == "Left"
    assert label_name(2) == "Right"
    assert label_name(1, zh=True) == "左手"
    assert label_name(None) is None


def test_normalize_p_three():
    assert normalize_p_three([0.1, 0.7, 0.2]) == [0.1, 0.7, 0.2]
    assert normalize_p_three(None) is None
    assert normalize_p_three([0.5, 0.5]) is None


def test_attach_judge_names():
    out = attach_judge_names(
        {"pred": 1, "p_three": [0.1, 0.8, 0.1], "t_rel": 3.6},
        label=1,
    )
    assert out["pred_name"] == "Left"
    assert out["correct"] is True
    assert out["p_three"] == [0.1, 0.8, 0.1]

    bad = attach_judge_names({"pred": 2, "p_three": [0.2, 0.2, 0.6]}, label=1)
    assert bad["correct"] is False
    assert bad["pred_name"] == "Right"
