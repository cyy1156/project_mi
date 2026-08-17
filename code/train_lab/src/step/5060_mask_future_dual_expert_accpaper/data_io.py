"""数据解析：A0=旧 hop100；A1+=pf1000 新臂。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _paths import PRE
from shared_hparams import DATA_TAG_A0, DATA_TAG_PF


def resolve_data_dir(tag: str) -> tuple[Path, str]:
    """返回 (dir, npy_prefix)。"""
    tag = tag.strip().lower()
    if tag == DATA_TAG_A0:
        return PRE / "out" / DATA_TAG_A0, "openbmi"
    if tag == DATA_TAG_PF:
        return PRE / "out" / DATA_TAG_PF, "openbmi"
    # 兼容：允许直接目录名
    d = PRE / "out" / tag
    return d, "openbmi"


def load_arrays(tag: str) -> dict:
    data_dir, prefix = resolve_data_dir(tag)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"A0 需要旧臂 out/{DATA_TAG_A0}；"
            f"A1+ 需要新臂 out/{DATA_TAG_PF}（见数据切片与边界过滤说明.md）"
        )

    def _load(name: str):
        p = data_dir / f"{prefix}_{name}.npy"
        if not p.is_file():
            raise FileNotFoundError(p)
        return np.load(p, mmap_mode="r")

    out = {
        "dir": data_dir,
        "prefix": prefix,
        "y_three": np.asarray(_load("y_three")),
        "subjects": np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True),
        "trial_id": np.asarray(_load("trial_id")),
    }
    # X：优先 X_full；否则 X
    xf = data_dir / f"{prefix}_X_full.npy"
    x0 = data_dir / f"{prefix}_X.npy"
    if xf.is_file():
        out["X_full"] = np.load(xf, mmap_mode="r")
        xm = data_dir / f"{prefix}_X_mask.npy"
        if xm.is_file():
            out["X_mask"] = np.load(xm, mmap_mode="r")
    elif x0.is_file():
        out["X"] = np.load(x0, mmap_mode="r")
    else:
        raise FileNotFoundError(f"未找到 {prefix}_X*.npy under {data_dir}")

    t0p = data_dir / f"{prefix}_t0_sec.npy"
    if t0p.is_file():
        out["t0_sec"] = np.asarray(np.load(t0p))
    return out


def to_bct(x: np.ndarray) -> np.ndarray:
    """统一到 (N, C, T)。"""
    a = np.asarray(x)
    if a.ndim == 4:
        # (N,1,C,T) or (N,C,T,1)
        if a.shape[1] == 1:
            a = a[:, 0]
        elif a.shape[-1] == 1:
            a = a[..., 0]
    if a.ndim != 3:
        raise ValueError(f"expect 3D after squeeze, got {a.shape}")
    # (N,T,C) → (N,C,T)
    if a.shape[1] in (500, 600, 1000) and a.shape[2] == 8:
        a = np.transpose(a, (0, 2, 1))
    return a
