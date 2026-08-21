"""方案 21 · 数据过滤与 pf800_mi080 几何（由 pf1000 v3 派生）。"""
from __future__ import annotations

import numpy as np

VIS_RAW_PTS = 600
MI080_TOTAL_PTS = 800
MI080_FUTURE_PTS = 200


def filter_indices_by_t0(
    idx: np.ndarray,
    t0_sec: np.ndarray | None,
    *,
    t0_max: float,
    t0_min: float = 0.4,
) -> np.ndarray:
    """保留 t0 ∈ [t0_min, t0_max] 的窗索引。"""
    if t0_sec is None:
        raise RuntimeError("scheme21 过滤需要 openbmi_t0_sec.npy")
    idx = np.asarray(idx, dtype=np.int64)
    t0 = np.asarray(t0_sec, dtype=np.float32)[idx]
    keep = (t0 >= float(t0_min) - 1e-6) & (t0 <= float(t0_max) + 1e-6)
    return idx[keep]


def crop_pf_mi080(x: np.ndarray) -> np.ndarray:
    """1000pt → 800pt：past100+cur500+future 前 200pt。"""
    x = np.asarray(x)
    if x.shape[-1] == MI080_TOTAL_PTS:
        return x
    if x.shape[-1] < MI080_TOTAL_PTS:
        raise ValueError(f"期望 T>=800，得到 {x.shape}")
    return x[..., :MI080_TOTAL_PTS].copy()


def scheme21_mask_future_pts(arm_extra: dict) -> int:
    if arm_extra.get("pf_mi080"):
        return MI080_FUTURE_PTS
    return 400
