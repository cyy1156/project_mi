"""按 mi_start / rest_start 切 4 s 窗，并做基线校正。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.signal import resample

from experiment_game.offline.load_session import (
    SessionEEG,
    rejected_trial_ids,
    time_to_sample,
)


@dataclass
class WindowSpec:
    kind: str  # "mi" | "rest"
    trial_id: int
    label: int  # 0/1/2 for y_three source
    t_start: float
    sample: int


def collect_window_specs(
    session: SessionEEG,
    *,
    phases: Optional[Sequence[str]] = ("acquire",),
) -> List[WindowSpec]:
    """
    从 events 收集训练窗起点。
    默认只要 phase∈acquire；排除 trial_reject。
    """
    reject = rejected_trial_ids(session.events)
    phase_set = set(phases) if phases is not None else None
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
        specs.append(
            WindowSpec(
                kind="mi" if ev == "mi_start" else "rest",
                trial_id=tid_i,
                label=lab,
                t_start=t0,
                sample=sample,
            )
        )
    return specs


def cut_window_with_baseline(
    x: np.ndarray,
    start: int,
    fs: float,
    *,
    dur_s: float = 4.0,
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


def resample_to_1000(x_win: np.ndarray, fs_in: float, fs_out: float = 250.0) -> np.ndarray:
    n_out = int(4.0 * fs_out)
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
    arr = np.stack(trials, axis=0)  # (N, 1000, 8)
    arr = np.transpose(arr, (0, 2, 1))  # (N, 8, 1000)
    return arr[:, None, :, :].astype(np.float32)


def labels_from_spec(spec: WindowSpec) -> tuple[int, int]:
    """→ (y_task, y_three)。"""
    if spec.kind == "rest" or spec.label == 0:
        return 0, 0
    return 1, int(spec.label)
