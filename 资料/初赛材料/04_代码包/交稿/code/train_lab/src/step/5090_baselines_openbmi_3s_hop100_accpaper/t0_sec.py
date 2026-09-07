"""3s/hop100 窗级 t0（秒）：trial 内按时间序映射到 [0.4, 1.0]（方案 24 · T/V）。"""

from __future__ import annotations

import numpy as np

T0_MIN = 0.4
T0_MAX = 1.0


def compute_window_t0_sec(
    trial_ids: np.ndarray,
    *,
    t0_min: float = T0_MIN,
    t0_max: float = T0_MAX,
) -> np.ndarray:
    """
    同一 trial_id 内窗顺序即 hop100 时间序；MI/Rest 均按 trial 内序号线性映射。
    单窗 trial → t0_min。
    """
    trial_ids = np.asarray(trial_ids, dtype=np.int64).reshape(-1)
    n = len(trial_ids)
    out = np.empty(n, dtype=np.float32)
    span = float(t0_max - t0_min)
    i = 0
    while i < n:
        tid = trial_ids[i]
        j = i + 1
        while j < n and trial_ids[j] == tid:
            j += 1
        cnt = j - i
        if cnt <= 1:
            out[i:j] = t0_min
        else:
            for k in range(cnt):
                out[i + k] = t0_min + span * (k / (cnt - 1))
        i = j
    return out


def t0_train_weight(t0_sec: np.ndarray, alpha: float) -> np.ndarray:
    """w(t0)=1−α·(t0−0.4)/0.6；α=0 → 全 1。"""
    if alpha <= 0.0:
        return np.ones_like(t0_sec, dtype=np.float32)
    t0 = np.asarray(t0_sec, dtype=np.float64)
    norm = (t0 - T0_MIN) / max(T0_MAX - T0_MIN, 1e-9)
    w = 1.0 - float(alpha) * norm
    return np.clip(w, 1e-6, None).astype(np.float32)


def t0_norm(t0_sec: np.ndarray) -> np.ndarray:
    """[0.4,1.0] → [0,1]。"""
    t0 = np.asarray(t0_sec, dtype=np.float64)
    return ((t0 - T0_MIN) / max(T0_MAX - T0_MIN, 1e-9)).astype(np.float32)
