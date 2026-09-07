"""OpenBMI 源域参考协方差（ref=src）缓存。"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from config import (
    CODE_ROOT,
    EA_REF_SRC_CACHE,
    OPENBMI_MAT_GLOB,
    OPENBMI_NOZ_DIR,
)
from ea import spatial_cov_avg


def _avg_cov_from_x_mmap(x_path: Path, *, stride: int = 8) -> np.ndarray:
    x = np.load(x_path, mmap_mode="r")
    n = len(x)
    if n == 0:
        raise ValueError(f"{x_path}: 空数组")
    idx = np.arange(0, n, max(1, stride), dtype=int)
    return spatial_cov_avg(x[idx])


def _avg_cov_from_mats(mat_glob: Path, *, limit: int | None = None) -> np.ndarray:
    """逐 mat 预处理（zscore=False）在线累计协方差，避免全量 noz merge。"""
    import sys

    pre_root = CODE_ROOT / "preprocess_lab"
    if str(pre_root) not in sys.path:
        sys.path.insert(0, str(pre_root))
    from src.datasets.openbmi.pipeline import preprocess_file_3s_hop100  # noqa: WPS433

    paths = sorted(glob.glob(str(mat_glob)))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"未找到 OpenBMI mat: {mat_glob}")
    acc = np.zeros((8, 8), dtype=np.float64)
    n_win = 0
    for mp in paths:
        x, *_rest, stats = preprocess_file_3s_hop100(mp, zscore=False)
        nw = int(stats.get("n_windows") or len(x))
        if nw == 0:
            continue
        cov = spatial_cov_avg(x, max_windows=min(200, len(x)))
        acc += cov * nw
        n_win += nw
    if n_win == 0:
        raise RuntimeError("OpenBMI mat 无有效窗")
    return acc / n_win


def load_ref_cov_src(
    *,
    rebuild: bool = False,
    smoke: bool = False,
) -> tuple[np.ndarray, dict]:
    """返回 (8,8) 参考协方差与 meta。"""
    meta: dict = {}
    if EA_REF_SRC_CACHE.is_file() and not rebuild:
        r = np.load(EA_REF_SRC_CACHE)
        meta["source"] = "cache"
        meta["path"] = str(EA_REF_SRC_CACHE)
        return r, meta

    noz_x = OPENBMI_NOZ_DIR / "openbmi_X.npy"
    if noz_x.is_file():
        r = _avg_cov_from_x_mmap(noz_x, stride=16 if not smoke else 64)
        meta["source"] = "openbmi_3s_hop100_noz"
        meta["path"] = str(noz_x)
    else:
        r = _avg_cov_from_mats(
            OPENBMI_MAT_GLOB,
            limit=4 if smoke else None,
        )
        meta["source"] = "openbmi_mats_online"
        meta["path"] = str(OPENBMI_MAT_GLOB)
        meta["note"] = (
            "未找到 openbmi_3s_hop100_noz；由 mat 在线估计。"
            "正式跑前建议: cd preprocess_lab && "
            "python -m src.datasets.openbmi.batch_3s_hop100 --no-zscore"
        )

    EA_REF_SRC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EA_REF_SRC_CACHE, r.astype(np.float64))
    meta["cache"] = str(EA_REF_SRC_CACHE)
    return r, meta
