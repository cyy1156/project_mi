"""游戏会话在线质量门控（P2）：同 trial REST 基线 → ERD/laterality。

质量分必须在 **无窗内 z-score** 的窗上计算；推理仍用 z-score 流。
通道序以 session 为准（游戏常见：C3,C4,Cz,...）。
"""

from __future__ import annotations

import numpy as np

# 与 teachable_v1 / P1-G2 对齐
MU_ERD_OK = -15.0
LATERALITY_OK = 8.0
MU_HZ = (8.0, 13.0)


def _channel_index(ch_names: list[str], name: str) -> int:
    try:
        return [str(c) for c in ch_names].index(name)
    except ValueError as e:
        raise KeyError(f"通道 {name} 不在 {ch_names}") from e


def _mu_power_ct(win_ct: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """(C,T) → Mu 功率 (C,)。"""
    x = np.asarray(win_ct, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(x.shape)
    t = x.shape[-1]
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    m = (freqs >= MU_HZ[0]) & (freqs < MU_HZ[1])
    if not np.any(m):
        return np.full(x.shape[0], 1e-12)
    return spec[:, m].mean(axis=-1) + 1e-12


def _erd(p_task: float, p_rest: float) -> float:
    return 100.0 * (p_task - p_rest) / (p_rest + 1e-12)


def build_gate_keeps(
    X_noz: np.ndarray,
    y_three: np.ndarray,
    trial_ids: np.ndarray,
    segs: np.ndarray,
    *,
    ch_names: list[str],
    top_p: float = 0.5,
    modes: tuple[str, ...] = ("H0", "H1", "H2", "H3"),
) -> dict[str, np.ndarray]:
    """返回与窗等长的 keep mask。

    - REST 段：H0–H3 一律保留
    - MI 段：按模式过滤；无 REST 基线时 H1/H2 下该 trial 全部 MI 不保留
    """
    n = len(y_three)
    y_three = np.asarray(y_three).astype(int)
    trial_ids = np.asarray(trial_ids).astype(np.int64)
    segs = np.asarray([str(s) for s in segs])
    assert len(X_noz) == n

    c3 = _channel_index(ch_names, "C3")
    c4 = _channel_index(ch_names, "C4")

    # per-window mu C3/C4
    mu3 = np.empty(n, dtype=np.float64)
    mu4 = np.empty(n, dtype=np.float64)
    for i in range(n):
        w = X_noz[i]
        if w.ndim == 3:
            w = w[0]
        band = _mu_power_ct(w)
        mu3[i] = float(band[c3])
        mu4[i] = float(band[c4])

    # rest baseline per trial
    rest_base: dict[int, tuple[float, float]] = {}
    for tid in np.unique(trial_ids):
        tid = int(tid)
        m = (trial_ids == tid) & (segs == "rest")
        if not np.any(m):
            continue
        rest_base[tid] = (float(mu3[m].mean()), float(mu4[m].mean()))

    # per-window laterality / erd_contra for MI
    lat = np.full(n, np.nan, dtype=np.float64)
    erd_contra = np.full(n, np.nan, dtype=np.float64)
    has_rest = np.zeros(n, dtype=bool)
    for i in range(n):
        if segs[i] != "mi":
            continue
        tid = int(trial_ids[i])
        if tid not in rest_base:
            continue
        has_rest[i] = True
        r3, r4 = rest_base[tid]
        e3 = _erd(mu3[i], r3)
        e4 = _erd(mu4[i], r4)
        lab = int(y_three[i])
        if lab == 1:  # left → contra C4
            erd_c, erd_i = e4, e3
        else:  # right → contra C3
            erd_c, erd_i = e3, e4
        erd_contra[i] = erd_c
        lat[i] = erd_i - erd_c

    is_rest = segs == "rest"
    is_mi = segs == "mi"
    out: dict[str, np.ndarray] = {}

    if "H0" in modes:
        out["H0"] = np.ones(n, dtype=bool)

    if "H1" in modes:
        # G2 同阈值
        ok = (
            is_mi
            & has_rest
            & (erd_contra <= MU_ERD_OK)
            & (lat >= LATERALITY_OK)
        )
        out["H1"] = is_rest | ok

    if "H2" in modes:
        ok = is_mi & has_rest & (lat >= LATERALITY_OK)
        out["H2"] = is_rest | ok

    if "H3" in modes:
        keep = is_rest.copy()
        for tid in np.unique(trial_ids):
            tid = int(tid)
            idxs = np.flatnonzero((trial_ids == tid) & is_mi)
            if len(idxs) == 0:
                continue
            # 无 rest 基线：保守全拒（与 H1 一致）
            if tid not in rest_base:
                continue
            scores = lat[idxs]
            # nan → 最低
            scores = np.where(np.isfinite(scores), scores, -1e18)
            k = max(1, int(np.ceil(len(scores) * float(top_p))))
            order = np.argsort(-scores)
            keep[idxs[order[:k]]] = True
        out["H3"] = keep

    return out


def gate_stats(keep: np.ndarray, segs: np.ndarray, seg_keys: np.ndarray) -> dict:
    segs = np.asarray([str(s) for s in segs])
    seg_keys = np.asarray([str(s) for s in seg_keys])
    keep = np.asarray(keep, dtype=bool)
    n_mi = int((segs == "mi").sum())
    n_rest = int((segs == "rest").sum())
    # segment abstain: MI segments with 0 kept windows
    mi_keys = sorted({k for k, s in zip(seg_keys.tolist(), segs.tolist()) if s == "mi"})
    n_mi_seg = len(mi_keys)
    n_mi_abstain = 0
    for k in mi_keys:
        m = seg_keys == k
        if int(keep[m].sum()) == 0:
            n_mi_abstain += 1
    return {
        "n_windows": int(len(keep)),
        "n_kept": int(keep.sum()),
        "n_mi_windows": n_mi,
        "n_rest_windows": n_rest,
        "n_mi_kept": int((keep & (segs == "mi")).sum()),
        "n_mi_segments": n_mi_seg,
        "n_mi_segments_abstain": n_mi_abstain,
        "mi_segment_abstain_rate": float(n_mi_abstain / n_mi_seg) if n_mi_seg else 0.0,
    }
