"""OpenBMI 原始时域：(N,1,8,500) → 磁盘 float16 mmap (N,8,500)，避免 5GB+ 常驻 RAM。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

N_TIMES_2S = 500
_CHUNK = 2048

_STEP = Path(__file__).resolve().parent.parent
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))


def openbmi_raw8_f16_path() -> Path:
    from data_paths import resolve_data

    data_dir, prefix = resolve_data("openbmi_2s_hop100")
    return Path(data_dir) / f"{prefix}_X_raw8_f16.npy"


def load_openbmi_raw8_f16(*, must_exist: bool = False) -> np.ndarray | None:
    """只打开 float16 缓存；不存在则返回 None（绝不先开 5GB float32）。"""
    out_path = openbmi_raw8_f16_path()
    if not out_path.is_file():
        if must_exist:
            raise FileNotFoundError(out_path)
        return None
    X = np.load(out_path, mmap_mode="r")
    print(f"[raw] mmap {out_path.name} {tuple(X.shape)} {X.dtype}", flush=True)
    return X


def squeeze_raw_2s_openbmi(X: np.ndarray | None = None) -> np.ndarray:
    """
    返回 (N,8,500) float16 memmap。
    优先直接复用 `openbmi_X_raw8_f16.npy`（16GB 机勿先开 float32 源）。
    """
    from data_paths import resolve_data

    data_dir, prefix = resolve_data("openbmi_2s_hop100")
    out_path = Path(data_dir) / f"{prefix}_X_raw8_f16.npy"

    # 16GB / Windows：有缓存就绝不再 np.load 5GB float32（待机缓存会吃光 Available）
    if X is None:
        cached = load_openbmi_raw8_f16()
        if cached is not None:
            return cached
        x_npy = Path(data_dir) / f"{prefix}_X.npy"
        print(f"[raw] no f16 cache; mmap float32 source once to build → {out_path.name}", flush=True)
        X = np.load(x_npy, mmap_mode="r")

    X = np.asarray(X)
    if X.ndim == 3 and X.shape[1] == 8 and X.shape[2] == N_TIMES_2S:
        if out_path.is_file() and int(np.load(out_path, mmap_mode="r").shape[0]) == len(X):
            return np.load(out_path, mmap_mode="r")
        return X

    if not (X.ndim == 4 and X.shape[1] == 1 and X.shape[2] == 8):
        raise ValueError(f"unexpected X shape for raw squeeze: {X.shape}")
    assert int(X.shape[3]) == N_TIMES_2S, X.shape

    n = int(X.shape[0])
    if out_path.is_file():
        cached = np.load(out_path, mmap_mode="r")
        if cached.shape == (n, 8, N_TIMES_2S):
            print(f"[raw] reuse cache {out_path.name} {cached.shape}", flush=True)
            return cached

    print(f"[raw] writing float16 squeeze → {out_path} …", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".npy.tmp")
    if tmp.exists():
        tmp.unlink()
    fp = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float16, shape=(n, 8, N_TIMES_2S)
    )
    for s in range(0, n, _CHUNK):
        e = min(s + _CHUNK, n)
        block = np.asarray(X[s:e, 0, :, :], dtype=np.float32)
        fp[s:e] = block.astype(np.float16, copy=False)
    fp.flush()
    del fp
    if out_path.exists():
        out_path.unlink()
    tmp.replace(out_path)
    print(f"[raw] squeeze done {out_path.name}", flush=True)
    return np.load(out_path, mmap_mode="r")
