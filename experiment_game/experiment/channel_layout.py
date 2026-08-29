"""Cyton 8 通道：全局设备序 + 模型权重轴 permute。

2026-08-29 实现平移至 experiment_game/core/channel_layout.py；
本文件仅为兼容转发（旧 import 路径继续可用），新代码请直接 import core。
"""

from experiment_game.core.channel_layout import (  # noqa: F401
    CHANNEL_ORDER,
    DEFAULT_CHANNEL_LABELS,
    DEVICE_CHANNEL_LABELS,
    DEVICE_TO_FROZEN,
    DEVICE_TO_MODEL_INPUT,
    FROZEN_CHANNEL_ORDER,
    MODEL_INPUT_CHANNEL_ORDER,
    permute_ch_time_to_model,
    reorder_device_to_frozen,
    reorder_device_to_model_input,
    reorder_frozen_to_device,
    reorder_model_input_to_device,
)

__all__ = [
    "CHANNEL_ORDER",
    "DEFAULT_CHANNEL_LABELS",
    "DEVICE_CHANNEL_LABELS",
    "DEVICE_TO_FROZEN",
    "DEVICE_TO_MODEL_INPUT",
    "FROZEN_CHANNEL_ORDER",
    "MODEL_INPUT_CHANNEL_ORDER",
    "permute_ch_time_to_model",
    "reorder_device_to_frozen",
    "reorder_device_to_model_input",
    "reorder_frozen_to_device",
    "reorder_model_input_to_device",
]
