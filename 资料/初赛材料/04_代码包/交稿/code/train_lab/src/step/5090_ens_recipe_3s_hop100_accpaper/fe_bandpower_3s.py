"""方案 26 · E2a 带功率 + 偏侧化特征（3s 窗）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

BANDS = ((8.0, 13.0), (13.0, 30.0))
CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
IX = {n: i for i, n in enumerate(CHANS)}
SFREQ = 250.0
_FE_DIM = 24
_CHUNK = 512


def _as_b8t(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    assert arr.ndim == 3 and arr.shape[1] == 8, arr.shape
    return arr.astype(np.float64, copy=False)


def bandpower_lateral_features(x8t: np.ndarray, *, sfreq: float = SFREQ) -> np.ndarray:
    """(B,8,T) → (B,24) float32：μ/β log 功率 + 偏侧 + 比值。"""
    x = _as_b8t(x8t)
    b, n_ch, _ = x.shape
    nyq = sfreq / 2.0
    coeffs = [butter(4, [lo / nyq, hi / nyq], btype="band") for lo, hi in BANDS]
    bp = np.empty((b, n_ch, len(BANDS)), dtype=np.float32)
    for start in range(0, b, _CHUNK):
        sl = slice(start, min(start + _CHUNK, b))
        block = np.array(x[sl], dtype=np.float64, copy=True)
        for bi, (b_f, a_f) in enumerate(coeffs):
            filt = filtfilt(b_f, a_f, block, axis=-1)
            power = np.mean(filt * filt, axis=-1)
            bp[sl, :, bi] = np.log(power + 1e-10).astype(np.float32)
    mu = bp[:, :, 0]
    beta = bp[:, :, 1]
    ratio = mu / (beta + 1e-6)
    lat_mu_c3c4 = mu[:, IX["C3"]] - mu[:, IX["C4"]]
    lat_mu_cp = mu[:, IX["CP3"]] - mu[:, IX["CP4"]]
    lat_beta_c3c4 = beta[:, IX["C3"]] - beta[:, IX["C4"]]
    lat_beta_cp = beta[:, IX["CP3"]] - beta[:, IX["CP4"]]
    lat_ratio_c3c4 = ratio[:, IX["C3"]] - ratio[:, IX["C4"]]
    flat = np.concatenate(
        [
            mu,
            beta,
            lat_mu_c3c4[:, None],
            lat_mu_cp[:, None],
            lat_beta_c3c4[:, None],
            lat_beta_cp[:, None],
            ratio[:, IX["C3"] : IX["C3"] + 1],
            ratio[:, IX["C4"] : IX["C4"] + 1],
            ratio.mean(axis=1, keepdims=True),
            lat_ratio_c3c4[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    assert flat.shape[1] == _FE_DIM, flat.shape
    return flat


def materialize_bandpower_cache(
    X_src: np.ndarray,
    out_path: Path,
    *,
    sfreq: float = SFREQ,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(X_src)
    if out_path.is_file():
        cached = np.load(out_path, mmap_mode="r")
        if cached.shape == (n, _FE_DIM):
            return out_path
    out = np.empty((n, _FE_DIM), dtype=np.float32)
    for start in range(0, n, _CHUNK):
        sl = slice(start, min(start + _CHUNK, n))
        block = np.array(X_src[sl], dtype=np.float32, copy=True)
        if block.ndim == 4:
            block = block[:, 0]
        out[sl] = bandpower_lateral_features(block, sfreq=sfreq)
    np.save(out_path, out)
    return out_path
