"""1 s 窗上的 2 频带 log 功率：(N,1,8,250) → (N,8,2)。"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

BANDS_HZ = ((8.0, 13.0), (13.0, 30.0))


def raw_to_bandpower(X: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    assert X.ndim == 3 and X.shape[1] == 8, X.shape
    n, n_ch, n_times = X.shape
    nyq = sfreq / 2.0
    out = np.empty((n, n_ch, len(BANDS_HZ)), dtype=np.float32)
    for bi, (lo, hi) in enumerate(BANDS_HZ):
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        flat = X.reshape(-1, n_times)
        filt = np.asarray([filtfilt(b, a, row) for row in flat], dtype=np.float64)
        power = np.mean(filt**2, axis=1).reshape(n, n_ch)
        out[:, :, bi] = np.log(power + 1e-10).astype(np.float32)
    return out
