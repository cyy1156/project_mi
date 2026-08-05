"""1 s / 40 ms 重叠滑窗（BCI2a / Stieger 共用）。"""
from __future__ import annotations

import numpy as np

from src.common.steps.resample_zscore import resample_to_1000, trial_zscore

FS_OUT = 250.0
WIN_SEC = 1.0
HOP_SEC = 0.04  # 40 ms
N_TIMES_1S = int(round(WIN_SEC * FS_OUT))  # 250
HOP_SAMPLES = int(round(HOP_SEC * FS_OUT))  # 10


def n_windows_for_duration(
    duration_sec: float,
    *,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
) -> int:
    """源时长 T 秒、T>=win_sec 时的理论窗数；否则 0。"""
    if duration_sec < win_sec - 1e-9:
        return 0
    return int(1 + round((duration_sec - win_sec) / hop_sec))


def slide_windows(
    x_tc: np.ndarray,
    *,
    fs: float = FS_OUT,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
) -> list[np.ndarray]:
    """
    x_tc: (T, C)。返回若干 (n_win, C)，尾部不足一窗丢弃（不 padding）。
    """
    n_win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    if hop <= 0 or n_win <= 0:
        raise ValueError(f"invalid win/hop: win={n_win}, hop={hop}")
    n = int(x_tc.shape[0])
    if n < n_win:
        return []
    outs: list[np.ndarray] = []
    start = 0
    while start + n_win <= n:
        outs.append(np.asarray(x_tc[start : start + n_win], dtype=np.float64))
        start += hop
    return outs


def slide_windows_1s(
    x_tc: np.ndarray,
    *,
    fs: float = FS_OUT,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
) -> list[np.ndarray]:
    """1 s / 40 ms 默认；参数可覆盖。"""
    return slide_windows(x_tc, fs=fs, win_sec=win_sec, hop_sec=hop_sec)


def extract_segment_baseline(
    x_tc: np.ndarray,
    t0: int,
    t1: int,
    fs: float,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    """截 [t0, t1)，优先用 t0 前 baseline_sec 做基线；不够则用段首 baseline_sec。"""
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


def segment_to_windows(
    seg_tc: np.ndarray,
    fs_in: float,
    *,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
    fs_out: float = FS_OUT,
    zscore: bool = True,
) -> list[np.ndarray]:
    """
    源段 (T_in, C) → 重采样到 fs_out → 滑窗 → 可选每窗 z-score。
    返回 list[(n_times, C)]，n_times = win_sec * fs_out。
    """
    if seg_tc is None or seg_tc.shape[0] < int(round(win_sec * fs_in)):
        return []
    dur = float(seg_tc.shape[0]) / float(fs_in)
    if dur < win_sec - 1e-9:
        return []
    seg_rs = resample_to_1000(seg_tc, fs_in=fs_in, fs_out=fs_out, win_sec=dur)
    wins = slide_windows(seg_rs, fs=fs_out, win_sec=win_sec, hop_sec=hop_sec)
    if zscore:
        return [trial_zscore(w) for w in wins]
    return wins


def segment_to_1s_windows(
    seg_tc: np.ndarray,
    fs_in: float,
    *,
    fs_out: float = FS_OUT,
    zscore: bool = True,
) -> list[np.ndarray]:
    """源段 → 40 ms 滑 1 s。"""
    return segment_to_windows(
        seg_tc, fs_in, win_sec=WIN_SEC, hop_sec=HOP_SEC, fs_out=fs_out, zscore=zscore
    )


def iter_rest_sources_cue_before(
    cue_samples: np.ndarray,
    fs: float,
    n_times: int,
    *,
    rest_sec: float = 4.0,
    task_sec: float = 4.0,
    min_win_sec: float | None = None,
) -> list[tuple[int, int]]:
    """
    Rest 源区间：每个 Cue（除首个之外）的 Cue 前 rest_sec。
    若与上一试次 MI [prev_cue, prev_cue+task_sec) 重叠，则缩短起点；
    缩短后仍不足 min_win_sec（默认 1 s）则丢弃该源段。
    返回 [(t0, t1), ...] 半开区间。
    """
    rest_len = int(round(rest_sec * fs))
    task_len = int(round(task_sec * fs))
    min_len = int(round((WIN_SEC if min_win_sec is None else min_win_sec) * fs))
    cues = np.sort(np.asarray(cue_samples, dtype=int).reshape(-1))
    out: list[tuple[int, int]] = []
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
