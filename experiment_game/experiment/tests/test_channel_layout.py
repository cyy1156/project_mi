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
    assert y[:, 1].mean() > 1.0
    assert y[:, 0].mean() < 1e-9


def test_cz_on_device_maps_to_frozen_cz():
    x = np.zeros((100, 8))
    cz_dev = DEVICE_CHANNEL_LABELS.index("CZ")
    x[:, cz_dev] = 40.0
    y = reorder_device_to_frozen(x)
    assert y[:, 0].mean() > 1.0


def test_fc4_cpz_device_slots_swapped():
    """设备 CH6=FC4、CH8=CPZ（帽位对调后）→ 冻结序 FC4@4、CPz@7。"""
    x = np.zeros((100, 8))
    fc4_dev = DEVICE_CHANNEL_LABELS.index("FC4")
    cpz_dev = DEVICE_CHANNEL_LABELS.index("CPZ")
    x[:, fc4_dev] = 30.0
    x[:, cpz_dev] = 20.0
    y = reorder_device_to_frozen(x)
    assert FROZEN_CHANNEL_ORDER[4] == "FC4"
    assert FROZEN_CHANNEL_ORDER[7] == "CPz"
    assert y[:, 4].mean() > 1.0
    assert y[:, 7].mean() > 1.0
    assert y[:, 4].mean() > y[:, 7].mean()
