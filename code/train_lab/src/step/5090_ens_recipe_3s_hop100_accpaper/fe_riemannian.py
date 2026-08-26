"""方案 26 · E2b Riemannian 切空间协方差特征。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _as_b8t(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    assert arr.ndim == 3 and arr.shape[1] == 8, arr.shape
    return arr.astype(np.float64, copy=False)


def cov_tangent_features(x8t: np.ndarray, *, band_split: bool = False) -> np.ndarray:
    """(B,8,T) → (B,36) or (B,72) 切空间向量。"""
    x = _as_b8t(x8t)
    b = x.shape[0]
    dim = 72 if band_split else 36
    out = np.empty((b, dim), dtype=np.float32)

    def _tangent_one(block: np.ndarray) -> np.ndarray:
        c = np.cov(block)
        c = c / (np.trace(c) + 1e-8)
        t = c - np.eye(8, dtype=np.float64) * (np.trace(c) / 8.0)
        v = t[np.triu_indices(8)]
        return v.astype(np.float32)

    for i in range(b):
        if not band_split:
            out[i] = _tangent_one(x[i])
        else:
            mid = x.shape[-1] // 2
            v1 = _tangent_one(x[i, :, :mid])
            v2 = _tangent_one(x[i, :, mid:])
            out[i] = np.concatenate([v1, v2])
    return out


def materialize_riemann_cache(X_src: np.ndarray, out_path: Path, *, band_split: bool = False) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(X_src)
    dim = 72 if band_split else 36
    if out_path.is_file():
        cached = np.load(out_path, mmap_mode="r")
        if cached.shape == (n, dim):
            return out_path
    out = np.empty((n, dim), dtype=np.float32)
    chunk = 256
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        block = np.array(X_src[sl], dtype=np.float32, copy=True)
        if block.ndim == 4:
            block = block[:, 0]
        out[sl] = cov_tangent_features(block, band_split=band_split)
    np.save(out_path, out)
    return out_path
