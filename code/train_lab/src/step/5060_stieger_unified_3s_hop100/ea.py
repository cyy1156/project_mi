"""欧氏对齐（EA）：前半估计 Σ_s，对齐到参考 Σ_ref，再窗内 z-score。"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def trial_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: (T, C) → 每通道独立标准化。"""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def _to_ct(w: np.ndarray) -> np.ndarray:
    """(1,8,T) 或 (8,T) → (C,T) float64。"""
    if w.ndim == 4:
        w = w[0, 0]
    elif w.ndim == 3:
        w = w[0]
    return np.asarray(w, dtype=np.float64)


def spatial_cov_avg(windows: np.ndarray, *, max_windows: int | None = None) -> np.ndarray:
    """对 (N,1,8,T) 或 (N,8,T) 估计平均空间协方差 (8,8)。"""
    n = len(windows)
    if n == 0:
        raise ValueError("spatial_cov_avg: 空输入")
    if max_windows is not None and n > max_windows:
        idx = np.linspace(0, n - 1, max_windows, dtype=int)
        windows = windows[idx]
    acc = np.zeros((8, 8), dtype=np.float64)
    for w in windows:
        ct = _to_ct(w)
        acc += ct @ ct.T / max(ct.shape[1], 1)
    return acc / len(windows)


def _sqrtm_psd(m: np.ndarray, *, inv: bool = False) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    m = 0.5 * (m + m.T)
    m = m + _EPS * np.eye(m.shape[0])
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, _EPS)
    if inv:
        w = 1.0 / w
    return (v * np.sqrt(w)) @ v.T


def ea_matrix(r_src: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    """W = R_ref^{1/2} R_src^{-1/2}。"""
    return _sqrtm_psd(r_ref) @ _sqrtm_psd(r_src, inv=True)


def apply_ea_window(w: np.ndarray, w_mat: np.ndarray) -> np.ndarray:
    """对单窗 (T,C) 或 (1,8,T) 施加 EA。"""
    if w.ndim == 4:
        ct = _to_ct(w).T  # (T,C)
        out = (ct @ w_mat.T).T
        return out.astype(np.float32)
    if w.ndim == 3:
        ct = _to_ct(w).T
        out = (ct @ w_mat.T).T
        return out.astype(np.float32)
    # (T,C)
    return (w @ w_mat.T).astype(np.float32)


def windows_to_model_tensor(windows_tc: list[np.ndarray]) -> np.ndarray:
    """list of (T,C) z-scored → (N,1,8,T)。"""
    arr = np.stack(windows_tc, axis=0)  # (N,T,C)
    arr = np.transpose(arr, (0, 2, 1))
    return arr[:, None, :, :].astype(np.float32)


def prepare_zscore_eval_batch(
    x_noz: np.ndarray,
    eval_idx: np.ndarray,
) -> np.ndarray:
    """无 EA：noz → 窗内 z-score → (N,1,8,T)。"""
    out: list[np.ndarray] = []
    for i in eval_idx:
        raw = _to_ct(x_noz[i]).T
        out.append(trial_zscore(raw))
    return windows_to_model_tensor(out)


def identity_ref_like(r_cal: np.ndarray) -> np.ndarray:
    """EA(cal) 严格口径：对齐到与 R_cal 迹同尺度的单位参考。"""
    n = int(r_cal.shape[0])
    scale = float(np.trace(r_cal) / max(n, 1))
    return np.eye(n, dtype=np.float64) * max(scale, _EPS)


def prepare_ea_eval_batch(
    x_noz: np.ndarray,
    cal_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    r_ref: np.ndarray,
    r_src: np.ndarray | None = None,
) -> np.ndarray:
    """对 eval 窗：EA(cal 估计的 R_s) → 窗内 z-score → (N,1,8,T)。"""
    if r_src is None:
        r_src = spatial_cov_avg(x_noz[cal_idx])
    w_mat = ea_matrix(r_src, r_ref)
    out: list[np.ndarray] = []
    for i in eval_idx:
        raw = _to_ct(x_noz[i]).T  # (T,C)
        aligned = apply_ea_window(raw, w_mat)
        out.append(trial_zscore(aligned))
    return windows_to_model_tensor(out)
