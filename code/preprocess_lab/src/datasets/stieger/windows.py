"""想象/反馈段取最后 4s + 基线校正（现行版；已替换「反馈起点起 4s」）。"""
from __future__ import annotations

import numpy as np


def feedback_start_index(time_ms: np.ndarray, feedback_t_ms: float = 2000.0) -> int:
    """
    time_ms: 1×nTime，相对目标呈现（ms）。
    返回第一个 t >= feedback_t_ms 的样本下标（0-based）。
    """
    t = np.asarray(time_ms).reshape(-1)
    idx = np.where(t >= feedback_t_ms - 1e-6)[0]
    if len(idx) == 0:
        # 回退：官方文档常用 4001（1-based）→ 0-based 4000
        return 4000
    return int(idx[0])


def feedback_end_index(
    n_times: int,
    resultind: int,
    fb0: int,
) -> int:
    """
    反馈终点（Python 半开区间右端）。
    官方 resultind 为 MATLAB 1-based 结束下标时，半开终点数值上等于 resultind。
    异常则回退 n_times。
    """
    end = int(resultind)
    end = min(max(end, fb0), n_times)
    if end <= fb0:
        end = n_times
    return int(end)


def extract_mi_or_rest_window(
    x_tc: np.ndarray,
    time_ms: np.ndarray,
    fs: float,
    *,
    resultind: int,
    feedback_t_ms: float = 2000.0,
    win_sec: float = 4.0,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    """
    x_tc: (n_times, n_ch)

    想象/反馈最长约 6s：分类窗取反馈段【最后 win_sec 秒】，
    即 [fb_end - win_sec, fb_end)。
    基线 = 分类窗起点前 baseline_sec。
    返回 (win_sec*fs, n_ch)；不够长则 None。

    满 6s 示意：
      反馈 [0 -------- 6]s
      基线          [1.5–2]
      分类窗           [2 ------ 6]  ← 最后 4s
    """
    fb0 = feedback_start_index(time_ms, feedback_t_ms)
    fb1 = feedback_end_index(x_tc.shape[0], resultind, fb0)
    n_base = int(round(baseline_sec * fs))
    n_win = int(round(win_sec * fs))

    if fb1 - fb0 < n_win:
        return None  # 反馈不足 4s

    win_start = fb1 - n_win
    base_start = win_start - n_base
    if base_start < 0:
        return None

    base = x_tc[base_start:win_start].mean(axis=0, keepdims=True)
    win = x_tc[win_start:fb1] - base
    if win.shape[0] != n_win:
        return None
    return win.astype(np.float64)