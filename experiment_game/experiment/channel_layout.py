"""Cyton 8 通道：设备序（LSL/CSV）↔ 冻结序（模型/UI）。"""

from __future__ import annotations

from typing import List

import numpy as np

# BrainFlow / OpenBCI 推流与 eeg.csv 表头顺序（CH1…CH8）
# CH6/CH8 帽位接反：设备序上 FC4 与 CPZ 对调（冻结序/模型轴不变）
DEVICE_CHANNEL_LABELS: List[str] = [
    "C3",
    "C4",
    "CZ",
    "CP3",
    "CP4",
    "FC4",
    "FC3",
    "CPZ",
]

# 比赛冻结序 = 模型输入轴 / 操作台热力图（索引 0=Cz, 1=C3, …）
FROZEN_CHANNEL_ORDER: List[str] = [
    "Cz",
    "C3",
    "C4",
    "CP3",
    "FC4",
    "FC3",
    "CP4",
    "CPz",
]

# frozen[:, i] = device[:, DEVICE_TO_FROZEN[i]]
DEVICE_TO_FROZEN: List[int] = [
    DEVICE_CHANNEL_LABELS.index(name.upper()) for name in FROZEN_CHANNEL_ORDER
]


def reorder_device_to_frozen(x: np.ndarray) -> np.ndarray:
    """(T, 8) 设备列序 → 冻结列序。"""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"期望 (T, 8)，得到 {arr.shape}")
    return arr[:, DEVICE_TO_FROZEN]


def reorder_frozen_to_device(x: np.ndarray) -> np.ndarray:
    """(T, 8) 冻结列序 → 设备列序（写 CSV 等少数场景）。"""
    arr = np.asarray(x, dtype=np.float64)
    inv = [0] * 8
    for fi, di in enumerate(DEVICE_TO_FROZEN):
        inv[di] = fi
    return arr[:, inv]
