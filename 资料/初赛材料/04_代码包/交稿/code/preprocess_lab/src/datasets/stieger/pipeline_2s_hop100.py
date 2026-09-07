"""Stieger 2 s / 100 ms（旁路） 协议（与旧「反馈末 2s/4s」流水线分离）。

Rest：Cue(=fb0) 前 4 s → slide_2s_hop100；不再使用 down 作 Rest。
Task：左/右反馈全程 [fb0, fb1) → slide_2s_hop100；丢 up/down、artifact、反馈 < 2 s。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.stieger.load_mat import load_stieger_mat, StiegerTrial
from src.datasets.stieger.windows import feedback_end_index, feedback_start_index
from src.common.steps.select_channels import select_channels, TARGET_CHANNELS
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.resample_zscore import to_model_tensor
from src.common.steps.slide_2s_hop100 import (
    N_TIMES_2S as N_TIMES_1S,
    WIN_SEC,
    extract_segment_baseline,
    segment_to_2s_hop100_windows as segment_to_1s_windows,
)

FS_OUT = 250.0
# 仅左/右；down/up 一律丢弃（Rest 只来自 Cue 前 4 s）
LR_TARGETS = {1, 2}
THREE_MAP = {2: 1, 1: 2}  # left→1, right→2


def _select_x(tr: StiegerTrial) -> np.ndarray | None:
    ch_names = [c.strip() for c in tr.ch_names]
    try:
        return select_channels(tr.x, ch_names)
    except KeyError:
        alias = {n.upper(): n for n in ch_names}
        mapped = []
        for want in TARGET_CHANNELS:
            key = want.upper()
            if key not in alias:
                return None
            mapped.append(ch_names.index(alias[key]))
        return tr.x[:, mapped]


def _append_wins(
    xs: list,
    y_task: list,
    y_three: list,
    trial_ids: list,
    wins: list[np.ndarray],
    lab_task: int,
    lab_three: int,
    tid: int,
) -> None:
    for w in wins:
        if w.shape != (N_TIMES_1S, 8):
            continue
        xs.append(w)
        y_task.append(int(lab_task))
        y_three.append(int(lab_three))
        trial_ids.append(int(tid))


def _process_one_trial_2s_hop100(
    tr: StiegerTrial,
    *,
    feedback_t_ms: float = 2000.0,
    baseline_sec: float = 0.5,
    rest_sec: float = 4.0,
    tid_base: int,
) -> tuple[list[np.ndarray], list[int], list[int], list[int], dict] | None:
    stats = {
        "n_task_wins": 0,
        "n_rest_wins": 0,
        "drop_reason": "",
    }
    if int(tr.artifact) == 1:
        stats["drop_reason"] = "artifact"
        return None
    if int(tr.targetnumber) not in LR_TARGETS:
        stats["drop_reason"] = "target"
        return None
    # task 1/2/3 中只要是左/右都保留（与旧批处理覆盖面一致，但不收 down）
    if int(tr.tasknumber) not in (1, 2, 3):
        stats["drop_reason"] = "task"
        return None

    x = _select_x(tr)
    if x is None:
        stats["drop_reason"] = "channels"
        return None
    x = car_reference(x)
    x = notch_and_bandpass(x, tr.fs)

    fb0 = feedback_start_index(tr.time_ms, feedback_t_ms)
    fb1 = feedback_end_index(x.shape[0], tr.resultind, fb0)
    min_task = int(round(WIN_SEC * tr.fs))
    if fb1 - fb0 < min_task:
        stats["drop_reason"] = "short_feedback"
        return None

    y_three = int(THREE_MAP[int(tr.targetnumber)])
    xs: list[np.ndarray] = []
    yt: list[int] = []
    y3: list[int] = []
    tids: list[int] = []

    # Task: 全反馈段
    task_seg = extract_segment_baseline(x, fb0, fb1, tr.fs, baseline_sec=baseline_sec)
    if task_seg is None:
        stats["drop_reason"] = "task_seg"
        return None
    task_wins = segment_to_1s_windows(task_seg, tr.fs)
    _append_wins(xs, yt, y3, tids, task_wins, 1, y_three, tid_base)
    stats["n_task_wins"] = len(task_wins)

    # Rest: Cue(=fb0) 前 4 s（可缩短到录音起点）
    rest_len = int(round(rest_sec * tr.fs))
    r0 = max(0, fb0 - rest_len)
    r1 = fb0
    if r1 - r0 >= min_task:
        rest_seg = extract_segment_baseline(x, r0, r1, tr.fs, baseline_sec=baseline_sec)
        if rest_seg is not None:
            rest_wins = segment_to_1s_windows(rest_seg, tr.fs)
            _append_wins(xs, yt, y3, tids, rest_wins, 0, 0, tid_base + 1)
            stats["n_rest_wins"] = len(rest_wins)

    if not xs:
        stats["drop_reason"] = "no_windows"
        return None
    return xs, yt, y3, tids, stats


def preprocess_session_2s_hop100(
    mat_path: Path | str,
    *,
    feedback_t_ms: float = 2000.0,
    baseline_sec: float = 0.5,
    rest_sec: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """单会话 → X, y_task, y_three, subjects, trial_id, stats。"""
    mat_path = Path(mat_path)
    trials = load_stieger_mat(mat_path)
    xs, yt, y3, sids, tids = [], [], [], [], []
    stats = {
        "n_raw": len(trials),
        "n_keep_trials": 0,
        "n_drop": 0,
        "n_task_wins": 0,
        "n_rest_wins": 0,
        "protocol": "slide_2s_hop100",
    }
    tid = 0
    for tr in trials:
        out = _process_one_trial_2s_hop100(
            tr,
            feedback_t_ms=feedback_t_ms,
            baseline_sec=baseline_sec,
            rest_sec=rest_sec,
            tid_base=tid,
        )
        if out is None:
            stats["n_drop"] += 1
            continue
        wins, y_task, y_three, trial_ids, st = out
        xs.extend(wins)
        yt.extend(y_task)
        y3.extend(y_three)
        tids.extend(trial_ids)
        sids.extend([tr.subject] * len(wins))
        stats["n_keep_trials"] += 1
        stats["n_task_wins"] += int(st["n_task_wins"])
        stats["n_rest_wins"] += int(st["n_rest_wins"])
        tid = max(trial_ids) + 1 if trial_ids else tid + 2

    if not xs:
        empty = np.zeros((0, 1, 8, N_TIMES_1S), np.float32)
        z = np.zeros((0,), np.int64)
        return empty, z, z.copy(), np.array([], dtype=object), z.copy(), stats

    X = to_model_tensor(xs)
    return (
        X,
        np.asarray(yt, dtype=np.int64),
        np.asarray(y3, dtype=np.int64),
        np.asarray(sids, dtype=object),
        np.asarray(tids, dtype=np.int64),
        stats,
    )


def sanity_check_outputs(X, y_task, y_three, n_times: int = N_TIMES_1S) -> None:
    assert len(X) > 0, "没有有效试次"
    assert X.ndim == 4 and X.shape[1:] == (1, 8, int(n_times)), X.shape
    assert len(X) == len(y_task) == len(y_three)
    assert set(np.unique(y_task)).issubset({0, 1})
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    assert np.all((y_three == 0) == (y_task == 0))
    assert np.all(y_task[y_three > 0] == 1)
    assert np.isfinite(X).all()
    print(
        "OK",
        "X",
        X.shape,
        "y_task",
        np.bincount(y_task, minlength=2),
        "y_three",
        np.bincount(y_three, minlength=3),
    )
