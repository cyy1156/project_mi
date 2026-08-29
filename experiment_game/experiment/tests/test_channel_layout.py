"""通道设备序 → 模型输入序映射单测。

2026-08-29 冻结：设备序与模型输入序统一为
FC3, C3, CP3, CZ, CPZ, FC4, C4, CP4（恒等映射）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiment_game.experiment.channel_layout import (  # noqa: E402
    CHANNEL_ORDER,
    DEVICE_CHANNEL_LABELS,
    MODEL_INPUT_CHANNEL_ORDER,
    permute_ch_time_to_model,
    reorder_device_to_model_input,
)

FROZEN_ORDER = [
    "FC3",
    "C3",
    "CP3",
    "CZ",
    "CPZ",
    "FC4",
    "C4",
    "CP4",
]


def test_channel_order_is_device_order():
    assert CHANNEL_ORDER == DEVICE_CHANNEL_LABELS
    assert CHANNEL_ORDER == FROZEN_ORDER


def test_model_order_equals_device_order():
    """2026-08-29 起两序必须完全一致。"""
    assert MODEL_INPUT_CHANNEL_ORDER == DEVICE_CHANNEL_LABELS == FROZEN_ORDER


def test_device_to_model_is_identity():
    x = np.arange(800, dtype=np.float64).reshape(100, 8)
    y = reorder_device_to_model_input(x)
    np.testing.assert_array_equal(x, y)


def test_each_channel_maps_to_same_model_index():
    for i, name in enumerate(FROZEN_ORDER):
        x = np.zeros((100, 8))
        x[:, i] = 50.0
        y = reorder_device_to_model_input(x)
        assert y[:, i].mean() > 1.0, f"{name} 应落在模型轴 {i}"
        for j in range(8):
            if j != i:
                assert y[:, j].mean() < 1e-9


def test_permute_ch_time_to_model_is_identity():
    x = np.arange(800, dtype=np.float64).reshape(8, 100)
    y = permute_ch_time_to_model(x)
    np.testing.assert_array_equal(x, y)
