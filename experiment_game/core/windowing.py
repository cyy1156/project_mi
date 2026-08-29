"""OpenBMI-Align 切窗权威实现（重构阶段 3）。

协议：Cue+0~4s 任务段 → 3s / hop100 滑窗 → (8, 750) float32；
Rest：Cue 前专用段或 cue-before 回退；Cue 前 0.5s 基线。

零项目依赖：仅标准库 + numpy + scipy（与 channel_layout 同层约定）。
windowing_version 用于溯源，改协议时递增。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample

from experiment_game.core.channel_layout import DEVICE_CHANNEL_LABELS

WINDOWING_VERSION = "openbmi_align_v1"
FS = 250.0
WIN_SEC = 3.0
HOP_SEC = 0.1
TASK_SEC = 4.0
BASELINE_BEFORE_CUE_S = 0.5
N_TIMES = int(round(WIN_SEC * FS))  # 750
HOP_SAMPLES = int(round(HOP_SEC * FS))  # 25
FROZEN: List[str] = list(DEVICE_CHANNEL_LABELS)


def lsl_to_sample(t_lsl: np.ndarray, t: float) -> int:
    return int(np.searchsorted(t_lsl, float(t)))


def cue_time_from_row(r: Dict[str, Any]) -> Optional[float]:
    tc, tm = r.get("t_cue"), r.get("t_mi_start")
    if tc not in (None, "") and tm not in (None, ""):
        tc_f, tm_f = float(tc), float(tm)
        if abs(tm_f - tc_f) <= 0.5:
            return tc_f
        return tm_f
    if tc not in (None, ""):
        return float(tc)
    if tm not in (None, ""):
        return float(tm)
    return None


def trial_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: (T, C) → 同形状，每通道独立标准化。"""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def resample_segment(
    x_win: np.ndarray,
    fs_in: float,
    *,
    fs_out: float = FS,
    win_sec: float,
) -> np.ndarray:
    n_out = int(round(win_sec * fs_out))
    if abs(fs_in - fs_out) < 1e-6 and x_win.shape[0] == n_out:
        return x_win.astype(np.float32)
    y = resample(x_win, n_out, axis=0)
    return np.asarray(y, dtype=np.float32)


def slide_windows(
    x_tc: np.ndarray,
    *,
    fs: float = FS,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
) -> List[np.ndarray]:
    """x_tc: (T, C) → list[(n_win, C)]，尾部不足一窗丢弃。"""
    n_win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    if hop <= 0 or n_win <= 0:
        raise ValueError(f"invalid win/hop: win={n_win}, hop={hop}")
    n = int(x_tc.shape[0])
    if n < n_win:
        return []
    outs: List[np.ndarray] = []
    start = 0
    while start + n_win <= n:
        outs.append(np.asarray(x_tc[start : start + n_win], dtype=np.float64))
        start += hop
    return outs


def segment_to_windows(
    seg_tc: np.ndarray,
    fs_in: float,
    *,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
    fs_out: float = FS,
    zscore: bool = True,
) -> List[np.ndarray]:
    if seg_tc is None or seg_tc.shape[0] < int(round(win_sec * fs_in)):
        return []
    dur = float(seg_tc.shape[0]) / float(fs_in)
    if dur < win_sec - 1e-9:
        return []
    seg_rs = resample_segment(seg_tc, fs_in, fs_out=fs_out, win_sec=dur)
    wins = slide_windows(seg_rs, fs=fs_out, win_sec=win_sec, hop_sec=hop_sec)
    if zscore:
        return [trial_zscore(w) for w in wins]
    return wins


def segment_to_3s_hop100_windows(
    seg_tc: np.ndarray,
    fs_in: float,
    *,
    fs_out: float = FS,
    zscore: bool = True,
) -> List[np.ndarray]:
    return segment_to_windows(
        seg_tc,
        fs_in,
        win_sec=WIN_SEC,
        hop_sec=HOP_SEC,
        fs_out=fs_out,
        zscore=zscore,
    )


def n_windows_for_duration(
    duration_sec: float,
    *,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
) -> int:
    if duration_sec < win_sec - 1e-9:
        return 0
    return int(1 + round((duration_sec - win_sec) / hop_sec))


def n_windows_3s_hop100(duration_sec: float) -> int:
    return n_windows_for_duration(duration_sec, win_sec=WIN_SEC, hop_sec=HOP_SEC)


def task_window_cue_0_to_4(
    x: np.ndarray,
    cue: int,
    fs: float,
    baseline_sec: float = BASELINE_BEFORE_CUE_S,
) -> Optional[np.ndarray]:
    """Cue 后 0~4s，Cue 前 baseline_sec 基线；(T,C) 或 None。"""
    n_win = int(round(TASK_SEC * fs))
    n_base = int(round(baseline_sec * fs))
    t0 = cue
    t1 = cue + n_win
    base_start = t0 - n_base
    if base_start < 0 or t1 > x.shape[0]:
        return None
    base = x[base_start:t0].mean(axis=0, keepdims=True)
    win = x[t0:t1] - base
    if win.shape[0] != n_win:
        return None
    return win.astype(np.float64)


def extract_segment_baseline(
    x_tc: np.ndarray,
    t0: int,
    t1: int,
    fs: float,
    baseline_sec: float = BASELINE_BEFORE_CUE_S,
) -> Optional[np.ndarray]:
    if t0 < 0 or t1 > x_tc.shape[0] or t1 <= t0:
        return None
    n_base = int(round(baseline_sec * fs))
    if n_base <= 0:
        return x_tc[t0:t1].astype(np.float64)
    if t0 - n_base >= 0:
        base = x_tc[t0 - n_base : t0].mean(axis=0, keepdims=True)
    else:
        if t1 - t0 < n_base:
            return None
        base = x_tc[t0 : t0 + n_base].mean(axis=0, keepdims=True)
    return (x_tc[t0:t1] - base).astype(np.float64)


def iter_rest_sources_cue_before(
    cue_samples: np.ndarray,
    fs: float,
    n_times: int,
    *,
    rest_sec: float = 4.0,
    task_sec: float = TASK_SEC,
    min_win_sec: Optional[float] = None,
) -> List[Tuple[int, int]]:
    rest_len = int(round(rest_sec * fs))
    task_len = int(round(task_sec * fs))
    min_len = int(round((WIN_SEC if min_win_sec is None else min_win_sec) * fs))
    cues = np.sort(np.asarray(cue_samples, dtype=int).reshape(-1))
    out: List[Tuple[int, int]] = []
    for i in range(1, len(cues)):
        end = int(cues[i])
        start = end - rest_len
        prev_task_end = int(cues[i - 1]) + task_len
        start = max(start, prev_task_end, 0)
        if end > n_times:
            continue
        if end - start < min_len:
            continue
        out.append((start, end))
    return out


def wins_to_model(wins: Sequence[np.ndarray]) -> List[np.ndarray]:
    """list[(T,C)] → list[(C,T)] float32，过滤非 3s 窗。"""
    n_need = int(round(WIN_SEC * FS))
    return [w.T.astype(np.float32) for w in wins if w.shape[0] == n_need]


def slide_3s_from_interval(
    x_tc: np.ndarray,
    t_lsl: np.ndarray,
    t_a: float,
    t_b: float,
    *,
    t0_min: float = 0.0,
    zscore: bool = True,
) -> List[np.ndarray]:
    """区间 [t_a+t0_min, t_b) 上 3s/hop100 → list[(8,750)]。"""
    if t_lsl.size == 0 or x_tc.size == 0:
        return []
    t0 = float(t_a) + float(t0_min)
    t1 = float(t_b)
    if t1 - t0 < WIN_SEC - 1e-9:
        return []
    i0 = lsl_to_sample(t_lsl, t0)
    i1 = lsl_to_sample(t_lsl, t1)
    if i1 <= i0:
        return []
    seg = x_tc[i0:i1]
    raw = segment_to_3s_hop100_windows(seg, FS, zscore=zscore)
    return wins_to_model(raw)


def to_nchw(wins_ct: Sequence[np.ndarray]) -> np.ndarray:
    """list[(C,T)] → (N,1,C,T)。"""
    if not wins_ct:
        return np.zeros((0, 1, 8, N_TIMES), dtype=np.float32)
    stacked = np.stack([np.asarray(w, dtype=np.float32) for w in wins_ct], axis=0)
    return stacked[:, None, :, :]
