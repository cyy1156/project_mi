"""连续 EEG 滤波：CAR + 陷波 50 Hz + 带通 8–30 Hz（scipy，对齐 preprocess_lab 意图）。"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def car_reference(x: np.ndarray) -> np.ndarray:
    """x: (n_times, n_ch)。"""
    return x - x.mean(axis=1, keepdims=True)


def notch_and_bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    """
    x: (n_times, n_ch)
    Notch 50 Hz + Bandpass 8–30 Hz，零相位 filtfilt。
    """
    data = np.asarray(x, dtype=np.float64)
    # notch
    b_n, a_n = iirnotch(w0=50.0, Q=30.0, fs=fs)
    # bandpass
    b_b, a_b = butter(N=4, Wn=[8.0, 30.0], btype="bandpass", fs=fs)
    out = np.empty_like(data)
    for c in range(data.shape[1]):
        y = filtfilt(b_n, a_n, data[:, c])
        out[:, c] = filtfilt(b_b, a_b, y)
    return out


def car_then_filter(x: np.ndarray, fs: float) -> np.ndarray:
    return notch_and_bandpass(car_reference(x), fs)
