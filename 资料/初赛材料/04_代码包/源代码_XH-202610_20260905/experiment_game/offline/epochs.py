"""按 mi_start / rest_start 切窗，并做基线校正。

历史 2s 固定窗路径（旧 phase / 兼容工具）；OpenBMI-Align 3s/hop100 权威实现见
`experiment_game.core.windowing` / `offline.openbmi_align_cut`。本期不统一合并本模块。

支持两种模式（与 preprocess_lab 对齐）：
- fixed：每阶段起点取 1 个固定窗（默认 2s → 500@250Hz）；
- slide：在阶段区间 [start, end) 内按窗长 + 步长滑动切多个窗。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.signal import resample

from experiment_game.core.channel_layout import DEVICE_TO_MODEL_INPUT
from experiment_game.offline.load_session import (
    SessionEEG,
    rejected_trial_ids,
    time_to_sample,
)

# 与 preprocess_lab / 训练统一
WIN_SEC = 2.0
FS_OUT = 250.0
N_TIMES = int(WIN_SEC * FS_OUT)  # 500
HOP_MS = 100.0
BASELINE_S = 0.5


@dataclass
class WindowSpec:
    kind: str  # "mi" | "rest"
    trial_id: int
    label: int  # 0/1/2 for y_three source
    t_start: float
    sample: int
    # 阶段闭合事件（mi_end / rest_end），滑窗范围上界；缺失为 None
    t_end: Optional[float] = None
    sample_end: Optional[int] = None


def collect_window_specs(
    session: SessionEEG,
    *,
    phases: Optional[Sequence[str]] = ("acquire",),
) -> List[WindowSpec]:
    """
    从 events 收集训练窗锚点（mi_start / rest_start），并配对闭合事件。
    默认只要 phase∈acquire；排除 trial_reject。
    """
    reject = rejected_trial_ids(session.events)
    phase_set = set(phases) if phases is not None else None
    # 先收 mi_end / rest_end，按 (event, trial_id) 索引
    ends: dict[tuple[str, int], dict] = {}
    for e in session.events:
        ev = e.get("event")
        if ev not in ("mi_end", "rest_end"):
            continue
        tid = e.get("trial_id")
        if tid is None:
            continue
        ends[(str(ev), int(tid))] = e

    specs: List[WindowSpec] = []
    for e in session.events:
        ev = e.get("event")
        if ev not in ("mi_start", "rest_start"):
            continue
        phase = e.get("phase")
        if phase_set is not None and phase not in phase_set:
            continue
        tid = e.get("trial_id")
        if tid is None:
            continue
        tid_i = int(tid)
        if tid_i in reject:
            continue
        label = e.get("label")
        if label is None:
            continue
        lab = int(label)
        if ev == "mi_start" and lab not in (1, 2):
            continue
        if ev == "rest_start" and lab != 0:
            continue
        t0 = float(e["t_lsl"])
        sample = time_to_sample(session.lsl_time, t0)
        t_end: Optional[float] = None
        sample_end: Optional[int] = None
        end_ev = ends.get(("mi_end" if ev == "mi_start" else "rest_end", tid_i))
        if end_ev is not None:
            t_end = float(end_ev["t_lsl"])
            # 录制未覆盖阶段结束（如会话末尾提前停录）时，
            # time_to_sample 会钳位到末样本、虚假压缩跨度；此时放弃跨度，
            # 退回单窗语义，由切窗环节以 window_oob 如实记录。
            if len(session.lsl_time) and t_end <= float(session.lsl_time[-1]) + 0.05:
                sample_end = time_to_sample(session.lsl_time, t_end)
        specs.append(
            WindowSpec(
                kind="mi" if ev == "mi_start" else "rest",
                trial_id=tid_i,
                label=lab,
                t_start=t0,
                sample=sample,
                t_end=t_end,
                sample_end=sample_end,
            )
        )
    return specs


def slide_offsets(
    span_samples: Optional[int],
    win_samples: int,
    hop_samples: int,
) -> List[int]:
    """
    滑窗起点偏移（相对阶段锚点）。span 未知时退化为单窗 [0]。
    窗必须完整落在阶段区间内，不越界进入下一阶段。
    """
    if span_samples is None:
        return [0]
    if win_samples > span_samples:
        return []
    hop = max(1, int(hop_samples))
    last = span_samples - win_samples
    return list(range(0, last + 1, hop))


def cut_window_with_baseline(
    x: np.ndarray,
    start: int,
    fs: float,
    *,
    dur_s: float = WIN_SEC,
    baseline_s: float = 0.5,
) -> Optional[np.ndarray]:
    """
    截 [start, start+dur)，用开头 baseline_s 均值减全窗。
    返回 (n_times, n_ch) 或越界 None。
    """
    n = int(round(dur_s * fs))
    if start < 0 or start + n > x.shape[0]:
        return None
    win = x[start : start + n, :].copy()
    b = max(1, int(round(baseline_s * fs)))
    win = win - win[:b, :].mean(axis=0, keepdims=True)
    return win


def resample_to_1000(
    x_win: np.ndarray,
    fs_in: float,
    fs_out: float = FS_OUT,
    win_sec: float = WIN_SEC,
) -> np.ndarray:
    """历史函数名；输出点数 = win_sec * fs_out（默认 2s→500）。"""
    n_out = int(round(win_sec * fs_out))
    if abs(fs_in - fs_out) < 1e-6 and x_win.shape[0] == n_out:
        return x_win.astype(np.float32)
    y = resample(x_win, n_out, axis=0)
    return np.asarray(y, dtype=np.float32)


def trial_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def to_model_tensor(trials: List[np.ndarray]) -> np.ndarray:
    arr = np.stack(trials, axis=0)  # (N, T, 8) 设备列序
    arr = arr[:, :, DEVICE_TO_MODEL_INPUT]  # → 模型训练列序
    arr = np.transpose(arr, (0, 2, 1))  # (N, 8, T)
    return arr[:, None, :, :].astype(np.float32)


def labels_from_spec(spec: WindowSpec) -> tuple[int, int]:
    """→ (y_task, y_three)。"""
    if spec.kind == "rest" or spec.label == 0:
        return 0, 0
    return 1, int(spec.label)
