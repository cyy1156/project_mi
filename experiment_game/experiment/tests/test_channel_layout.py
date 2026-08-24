"""通道设备序 → 冻结序映射单测。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiment_game.experiment.channel_layout import (  # noqa: E402
    DEVICE_CHANNEL_LABELS,
    FROZEN_CHANNEL_ORDER,
    reorder_device_to_frozen,
)


def test_c3_on_device_maps_to_frozen_c3_not_cz():
    """设备 CH1(C3) 扰动应落在冻结索引 1(C3)，而非 0(Cz)。"""
    x = np.zeros((100, 8))
    x[:, 0] = 50.0  # device index 0 = C3
    y = reorder_device_to_frozen(x)
    assert FROZEN_CHANNEL_ORDER[1] == "C3"
    assert y[:, 1].std() > 1.0
    assert y[:, 0].std() < 1e-9


def test_cz_on_device_maps_to_frozen_cz():
    x = np.zeros((100, 8))
    cz_dev = DEVICE_CHANNEL_LABELS.index("CZ")
    x[:, cz_dev] = 40.0
    y = reorder_device_to_frozen(x)
    assert y[:, 0].std() > 1.0
