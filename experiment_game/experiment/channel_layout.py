"""Cyton 8 通道：全局设备序 + 模型权重轴 permute。"""

from __future__ import annotations

from typing import List

import numpy as np

# BrainFlow / OpenBCI / eeg.csv / RingBuffer / UI / 推理展示 — 全局统一
# 2026-08-29 冻结：设备序与模型序统一为同一套（8 导新接线序）。
DEVICE_CHANNEL_LABELS: List[str] = [
    "FC3",
    "C3",
    "CP3",
    "CZ",
    "CPZ",
    "FC4",
    "C4",
    "CP4",
]

# 对外唯一通道序（与设备序相同）
CHANNEL_ORDER: List[str] = list(DEVICE_CHANNEL_LABELS)

# 模型训练/推理输入轴序：自 2026-08-29 起与设备序完全一致（无需 permute）。
# 保留本常量与 reorder/permute 函数以兼容旧调用方；映射现为恒等。
MODEL_INPUT_CHANNEL_ORDER: List[str] = list(DEVICE_CHANNEL_LABELS)


def _device_index(name: str) -> int:
    key = str(name or "").upper()
    for i, lab in enumerate(DEVICE_CHANNEL_LABELS):
        if lab.upper() == key:
            return i
    raise KeyError(f"通道 {name!r} 不在 DEVICE_CHANNEL_LABELS {DEVICE_CHANNEL_LABELS}")


# model[:, i] = device[:, DEVICE_TO_MODEL_INPUT[i]]
# 两序一致后为恒等映射 [0..7]；保留以兼容旧调用与未来重新分序。
DEVICE_TO_MODEL_INPUT: List[int] = [
    _device_index(n) for n in MODEL_INPUT_CHANNEL_ORDER
]

# 兼容旧名
FROZEN_CHANNEL_ORDER = MODEL_INPUT_CHANNEL_ORDER
DEVICE_TO_FROZEN = DEVICE_TO_MODEL_INPUT


def reorder_device_to_model_input(x: np.ndarray) -> np.ndarray:
    """(T, 8) 设备列序 → 模型训练列序（两序统一后为恒等拷贝）。"""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"期望 (T, 8)，得到 {arr.shape}")
    return arr[:, DEVICE_TO_MODEL_INPUT]


def reorder_model_input_to_device(x: np.ndarray) -> np.ndarray:
    """(T, 8) 模型训练列序 → 设备列序。"""
    arr = np.asarray(x, dtype=np.float64)
    inv = [0] * 8
    for mi, di in enumerate(DEVICE_TO_MODEL_INPUT):
        inv[di] = mi
    return arr[:, inv]


def permute_ch_time_to_model(x: np.ndarray) -> np.ndarray:
    """(8, T) 设备行序 → 模型训练行序（InferenceService forward 前）。"""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 8:
        raise ValueError(f"期望 (8, T)，得到 {arr.shape}")
    return arr[DEVICE_TO_MODEL_INPUT, :].copy()


# 兼容旧 API
reorder_device_to_frozen = reorder_device_to_model_input
reorder_frozen_to_device = reorder_model_input_to_device
