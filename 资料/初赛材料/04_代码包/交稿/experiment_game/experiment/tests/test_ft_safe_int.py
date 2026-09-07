"""FT 切窗：CSV 空 invalid/rejected → NaN 不得炸 int()。"""

from __future__ import annotations

import math

import numpy as np

from experiment_game.tools.ft_subject_from_v3 import _safe_int


def test_safe_int_nan_and_empty():
    assert _safe_int(float("nan"), 0) == 0
    assert _safe_int(np.nan, 0) == 0
    assert _safe_int(None, 0) == 0
    assert _safe_int("", 0) == 0
    assert _safe_int("  ", 0) == 0
    assert _safe_int(1, 0) == 1
    assert _safe_int(1.0, 0) == 1
    assert _safe_int("2", 0) == 2
    assert _safe_int(math.nan or 0, 0) == 0  # NaN is truthy; helper still safe
