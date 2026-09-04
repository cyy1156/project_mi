"""OpenBMI · 45ch 固定 3s 窗（Exp36 C1）。

协议：
  选 INTERSECT_45 → CAR → notch50 → bandpass 8–30
  Task：Cue 后 [0, 3)s（基线 Cue 前 0.5s）→ 重采样 250 → 窗内 z-score → (1,45,750)
  Rest：Cue 前 3s（可缩短）→ 同上
  标签：OpenBMI 约定 Rest=0, Left=1, Right=2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.eeg_types import ContinuousEEG
from src.common.steps.epoch_baseline import rest_window_with_baseline
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.resample_zscore import resample_to_1000, to_model_tensor, trial_zscore
from src.common.steps.slide_1s import iter_rest_sources_cue_before
from src.datasets.bci2a.labels import filter_left_right_events
from src.datasets.challenge_mi.channels_intersect import INTERSECT_45, indices_in_names
from src.datasets.openbmi.load_mat import load_openbmi_mat

WIN_SEC = 3.0
N_TIMES = 750  # 3s @ 250 Hz
N_CH = len(INTERSECT_45)
PROTOCOL = "openbmi_3s_fixed_45ch"


def select_45(x_tc: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """x (T,C) → (T,45) 官方交集序。"""
    idx = indices_in_names(ch_names, INTERSECT_45)
    return np.asarray(x_tc[:, idx], dtype=np.float64)


def task_window_cue_0_to_3(
    x: np.ndarray,
    cue: int,
    fs: float,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    n_win = int(round(WIN_SEC * fs))
    n_base = int(round(baseline_sec * fs))
    t0 = int(cue)
    t1 = t0 + n_win
    base_start = t0 - n_base
    if base_start < 0 or t1 > x.shape[0]:
        return None
    base = x[base_start:t0].mean(axis=0, keepdims=True)
    win = x[t0:t1] - base
    if win.shape[0] != n_win:
        return None
    return win.astype(np.float64)


def _append(
    xs: list,
    y_task: list,
    y_three: list,
    trial_ids: list,
    tid: int,
    win: np.ndarray | None,
    lab_task: int,
    lab_three: int,
    fs: float,
    *,
    zscore: bool,
) -> bool:
    if win is None:
        return False
    win = resample_to_1000(win, fs_in=fs, fs_out=250.0, win_sec=WIN_SEC)
    if win.shape != (N_TIMES, N_CH):
        return False
    if zscore:
        win = trial_zscore(win)
    xs.append(win)
    y_task.append(int(lab_task))
    y_three.append(int(lab_three))
    trial_ids.append(int(tid))
    return True


def preprocess_run_3s_fixed_45ch(
    eeg: ContinuousEEG,
    add_rest: bool = True,
    max_rest: int | None = None,
    *,
    zscore: bool = True,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = select_45(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs, l_freq=l_freq, h_freq=h_freq)
    kept = filter_left_right_events(eeg.events, eeg.artifacts)

    xs: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []
    trial_ids: list[int] = []
    tid = 0

    for cue, lab_task, lab_three, _ in kept:
        win = task_window_cue_0_to_3(x, int(cue), eeg.fs)
        if _append(xs, y_task, y_three, trial_ids, tid, win, int(lab_task), int(lab_three), eeg.fs, zscore=zscore):
            tid += 1

    if add_rest and len(kept) > 0:
        sources = iter_rest_sources_cue_before(
            kept[:, 0],
            eeg.fs,
            x.shape[0],
            rest_sec=3.0,
            task_sec=3.0,
            min_win_sec=WIN_SEC,
        )
        if max_rest is None:
            n_left = int(np.sum(kept[:, 2] == 1))
            n_right = int(np.sum(kept[:, 2] == 2))
            max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        sources = sources[: int(max_rest)]
        for t0, _t1 in sources:
            win = rest_window_with_baseline(
                x, int(t0), eeg.fs, win_sec=WIN_SEC, baseline_sec=0.5
            )
            if _append(xs, y_task, y_three, trial_ids, tid, win, 0, 0, eeg.fs, zscore=zscore):
                tid += 1

    if not xs:
        empty_x = np.zeros((0, 1, N_CH, N_TIMES), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy(), empty_y.copy()

    return (
        to_model_tensor(xs),
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )


def preprocess_file_3s_fixed_45ch(
    mat_path: Path | str,
    *,
    add_rest: bool = True,
    zscore: bool = True,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    mat_path = Path(mat_path)
    runs = load_openbmi_mat(mat_path)
    xs, yts, y3s, tids, sids = [], [], [], [], []
    tid_offset = 0
    stats = {
        "n_runs": len(runs),
        "n_windows": 0,
        "subject": runs[0].subject if runs else "",
        "protocol": PROTOCOL,
        "zscore": bool(zscore),
        "bandpass_hz": [float(l_freq), float(h_freq)],
        "blocks": ["EEG_MI_train"],
        "n_chans": N_CH,
        "channels": list(INTERSECT_45),
        "win_sec": WIN_SEC,
        "hop_sec": None,
        "task": "cue_0_to_3s",
        "rest": "cue_before_3s",
    }
    for eeg in runs:
        X, yt, y3, tid = preprocess_run_3s_fixed_45ch(
            eeg, add_rest=add_rest, zscore=zscore, l_freq=l_freq, h_freq=h_freq
        )
        if len(yt) == 0:
            continue
        tid = tid + tid_offset
        tid_offset = int(tid.max()) + 1
        xs.append(X)
        yts.append(yt)
        y3s.append(y3)
        tids.append(tid)
        sids.append(np.array([eeg.subject] * len(yt), dtype=object))

    if not xs:
        empty = np.zeros((0, 1, N_CH, N_TIMES), np.float32)
        z = np.zeros((0,), np.int64)
        return empty, z, z.copy(), np.array([], dtype=object), z.copy(), stats

    X = np.concatenate(xs, axis=0)
    y_task = np.concatenate(yts, axis=0)
    y_three = np.concatenate(y3s, axis=0)
    subjects = np.concatenate(sids, axis=0)
    trial_id = np.concatenate(tids, axis=0)
    stats["n_windows"] = int(len(y_task))
    stats["y_task"] = np.bincount(y_task, minlength=2).tolist()
    stats["y_three"] = np.bincount(y_three, minlength=3).tolist()
    return X, y_task, y_three, subjects, trial_id, stats
