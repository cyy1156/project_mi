"""OpenBMI · 2s/hop100 预处理流水线。

步骤（与方案 §预处理 一致）：
  1. 选 8 导（Cz,C3,C4,CP3,FC4,FC3,CP4,CPz）
  2. CAR → notch50 → bandpass 8–30（原始 fs，多为 1000 Hz）
  3. Task：Cue 后 0–4 s（Cue 前 0.5 s 基线校正）→ 2s/100ms 滑窗 → 重采样 250 + 窗内 z-score
  4. Rest：Cue 前 4 s（可缩短避让）→ 同上滑窗
  5. 标签：left→y_three=1，right→y_three=2，Rest→0；y_task=0/1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.eeg_types import ContinuousEEG
from src.common.steps.epoch_baseline import task_window_cue_0_to_4
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.resample_zscore import to_model_tensor
from src.common.steps.select_channels import select_channels
from src.common.steps.slide_1s import extract_segment_baseline, iter_rest_sources_cue_before
from src.common.steps.slide_2s_hop100 import (
    N_TIMES_2S,
    WIN_SEC as WIN_SEC_2S_HOP100,
    segment_to_2s_hop100_windows,
)
from src.datasets.bci2a.labels import filter_left_right_events
from src.datasets.openbmi.load_mat import load_openbmi_mat


def preprocess_run_2s_hop100(
    eeg: ContinuousEEG,
    add_rest: bool = True,
    max_rest: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """单段连续流 → X (N,1,8,500), y_task, y_three, trial_id（段内从 0 起）。"""
    x = select_channels(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs)

    kept = filter_left_right_events(eeg.events, eeg.artifacts)

    xs: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []
    trial_ids: list[int] = []
    tid = 0
    n_times = N_TIMES_2S

    for cue, lab_task, lab_three, _ in kept:
        seg = task_window_cue_0_to_4(x, int(cue), eeg.fs)
        if seg is None:
            continue
        wins = segment_to_2s_hop100_windows(seg, eeg.fs)
        if not wins:
            continue
        for w in wins:
            if w.shape != (n_times, 8):
                continue
            xs.append(w)
            y_task.append(int(lab_task))
            y_three.append(int(lab_three))
            trial_ids.append(tid)
        tid += 1

    if add_rest and len(kept) > 0:
        sources = iter_rest_sources_cue_before(
            kept[:, 0],
            eeg.fs,
            x.shape[0],
            rest_sec=4.0,
            task_sec=4.0,
            min_win_sec=WIN_SEC_2S_HOP100,
        )
        if max_rest is None:
            n_left = int(np.sum(kept[:, 2] == 1))
            n_right = int(np.sum(kept[:, 2] == 2))
            max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        sources = sources[: int(max_rest)]

        for t0, t1 in sources:
            seg = extract_segment_baseline(x, int(t0), int(t1), eeg.fs, baseline_sec=0.5)
            if seg is None:
                continue
            wins = segment_to_2s_hop100_windows(seg, eeg.fs)
            for w in wins:
                if w.shape != (n_times, 8):
                    continue
                xs.append(w)
                y_task.append(0)
                y_three.append(0)
                trial_ids.append(tid)
            if wins:
                tid += 1

    if not xs:
        empty_x = np.zeros((0, 1, 8, n_times), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy(), empty_y.copy()

    X = to_model_tensor(xs)
    return (
        X,
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )


def preprocess_file_2s_hop100(
    mat_path: Path | str,
    *,
    add_rest: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """单 mat（仅 EEG_MI_train 块）→ 切窗；subjects 全为 openbmi:subjNN。"""
    mat_path = Path(mat_path)
    runs = load_openbmi_mat(mat_path)
    xs, yts, y3s, tids, sids = [], [], [], [], []
    tid_offset = 0
    stats = {
        "n_runs": len(runs),
        "n_windows": 0,
        "subject": runs[0].subject if runs else "",
        "protocol": "openbmi_2s_hop100",
    }
    for eeg in runs:
        X, yt, y3, tid = preprocess_run_2s_hop100(eeg, add_rest=add_rest)
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
        empty = np.zeros((0, 1, 8, N_TIMES_2S), np.float32)
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


def sanity_check_outputs(X, y_task, y_three, n_times: int = N_TIMES_2S) -> None:
    assert len(X) > 0, "没有有效窗"
    assert X.ndim == 4 and X.shape[1:] == (1, 8, int(n_times)), X.shape
    assert len(X) == len(y_task) == len(y_three)
    assert set(np.unique(y_task)).issubset({0, 1})
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    assert np.all((y_three == 0) == (y_task == 0))
    assert np.all(y_task[y_three > 0] == 1)
    assert np.isfinite(X).all()
