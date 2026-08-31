"""finetune 门控与试次划分修复单测。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from experiment_game.pipeline.finetune import (  # noqa: E402
    RELEASE_HELDOUT_ACC_MIN,
    _heldout_acc_threshold,
    _trial_split,
    evaluate_release_gate,
)


def test_trial_split_raises_when_no_heldout():
    with pytest.raises(ValueError, match="heldout 窗为空"):
        _trial_split(np.array([], dtype=object), train_frac=0.7, seed=42)


def test_heldout_acc_threshold_balanced_data():
    y = np.array([0, 1, 2] * 100)
    th = _heldout_acc_threshold(y)
    assert th == RELEASE_HELDOUT_ACC_MIN


def test_heldout_acc_threshold_rest_heavy():
    y = np.array([0] * 61 + [1] * 20 + [2] * 19)
    th = _heldout_acc_threshold(y)
    assert th > 0.60


def test_evaluate_release_gate_uses_adaptive_min():
    y = np.array([0] * 61 + [1] * 20 + [2] * 19)
    rep = {
        "acc_after_heldout": 0.55,
        "acc_after_train": 0.70,
        "heldout_pred_dist": {
            "pred_counts": {0: 50, 1: 25, 2: 25},
            "max_class_frac": 0.50,
        },
    }
    gate = evaluate_release_gate(rep, y_heldout=y)
    assert gate["heldout_acc_min"] > 0.60
    assert gate["checks"]["heldout_acc"] is False
