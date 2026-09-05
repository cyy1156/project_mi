"""2 s 窗上的 2 频带 log 功率：(N,1,8,500) → (N,8,2)。

分块计算，避免一次性分配 (N*8, T) float64 导致 OOM。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

BANDS_HZ = ((8.0, 13.0), (13.0, 30.0))
_CHUNK = 2048


def raw_to_bandpower(X: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    """支持 mmap：只按块读入，输出 (N,8,n_band) float32（通常几十 MB）。"""
    # 勿 np.asarray 整表拷贝；对 memmap 用切片视图
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    assert X.ndim == 3 and X.shape[1] == 8, X.shape
    n, n_ch, n_times = X.shape
    nyq = sfreq / 2.0
    out = np.empty((n, n_ch, len(BANDS_HZ)), dtype=np.float32)
    coeffs = [butter(4, [lo / nyq, hi / nyq], btype="band") for lo, hi in BANDS_HZ]

    for start in range(0, n, _CHUNK):
        sl = slice(start, min(start + _CHUNK, n))
        block = np.array(X[sl], dtype=np.float64, copy=True)  # (B, 8, T)
        for bi, (b, a) in enumerate(coeffs):
            filt = filtfilt(b, a, block, axis=-1)
            power = np.mean(filt * filt, axis=-1)
            out[sl, :, bi] = np.log(power + 1e-10).astype(np.float32)
    return out
