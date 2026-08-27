"""BCI2a 22→8 通道映射（与 preprocess_lab 一致）。"""

from __future__ import annotations

from typing import List

import numpy as np

EEG22: List[str] = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

TARGET_CHANNELS: List[str] = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]


def select_8ch(x: np.ndarray, ch_names: List[str]) -> np.ndarray:
    """x: (n_times, n_ch) → (n_times, 8)。"""
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    missing = [c for c in TARGET_CHANNELS if c not in name_to_idx]
    if missing:
        raise KeyError(f"Missing channels: {missing}")
    idx = [name_to_idx[c] for c in TARGET_CHANNELS]
    return np.asarray(x[:, idx], dtype=np.float64)
