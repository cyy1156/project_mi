"""OpenBMI 原始时域：分块 mmap (N,1,8,500) → (N,8,500)，避免整表 float32 进内存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_TIMES_2S = 500
_CHUNK = 2048


def squeeze_raw_2s(
    X: np.ndarray,
    cache_path: Path | str | None = None,
    *,
    chunk: int = _CHUNK,
) -> np.ndarray:
    """
    支持源 X 为 memmap。若提供 cache_path 且文件已存在则直接 mmap 读；
    否则分块写入 cache_path（或仅当源已是 (N,8,T) 时返回视图/数组）。
    """
    if X.ndim == 3 and X.shape[1] == 8 and X.shape[2] == N_TIMES_2S:
        return X

    assert X.ndim == 4 and X.shape[1] == 1 and X.shape[2] == 8, X.shape
    assert int(X.shape[3]) == N_TIMES_2S, X.shape

    n = int(X.shape[0])
    cache: Path | None = Path(cache_path) if cache_path is not None else None

    if cache is not None and cache.is_file():
        out = np.load(cache, mmap_mode="r")
        assert out.shape == (n, 8, N_TIMES_2S), (out.shape, n)
        return out

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            cache.unlink()
        fp = np.lib.format.open_memmap(
            cache, mode="w+", dtype=np.float32, shape=(n, 8, N_TIMES_2S)
        )
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            fp[s:e] = np.array(X[s:e, 0, :, :], dtype=np.float32, copy=False)
        fp.flush()
        return fp

    out = np.empty((n, 8, N_TIMES_2S), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = np.array(X[s:e, 0, :, :], dtype=np.float32, copy=False)
    return out
