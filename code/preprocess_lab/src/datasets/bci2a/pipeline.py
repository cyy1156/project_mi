"""BCI2a 流水线：任务 = Cue 后 2~4s；静息 = Cue 前 2s → (N,1,8,500)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.eeg_types import ContinuousEEG
from src.datasets.bci2a.load_mat import load_bci2a_mat
from src.common.steps.select_channels import select_channels
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.datasets.bci2a.labels import (
    filter_left_right_events,
    extract_rest_cues,
)
from src.common.steps.epoch_baseline import (
    task_window_cue_2_to_4,
    task_window_cue_0_to_4,
    rest_window_with_baseline,
)
from src.common.steps.resample_zscore import (
    resample_to_1000,
    trial_zscore,
    to_model_tensor,
)
from src.common.steps.slide_1s import (
    N_TIMES_1S,
    extract_segment_baseline,
    iter_rest_sources_cue_before,
    segment_to_1s_windows,
)
from src.common.steps.slide_2s_hop100 import (
    N_TIMES_2S,
    WIN_SEC as WIN_SEC_2S_HOP100,
    HOP_SEC as HOP_SEC_2S_HOP100,
    segment_to_2s_hop100_windows,
)
from src.common.steps.slide_3s_hop100 import (
    N_TIMES_3S,
    WIN_SEC as WIN_SEC_3S_HOP100,
    segment_to_3s_hop100_windows,
)
from src.common.steps.split_subjects import split_all_trials

WIN_SEC = 2.0
N_TIMES = 500  # 2s @ 250Hz


def _append_window(
    xs: list,
    y_task: list,
    y_three: list,
    win: np.ndarray | None,
    lab_task: int,
    lab_three: int,
    fs: float,
) -> None:
    if win is None:
        return
    win = resample_to_1000(win, fs_in=fs, fs_out=250.0, win_sec=WIN_SEC)
    if win.shape != (N_TIMES, 8):
        return
    win = trial_zscore(win)
    xs.append(win)
    y_task.append(lab_task)
    y_three.append(lab_three)


def preprocess_run(
    eeg: ContinuousEEG,
    add_rest: bool = True,
    max_rest: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    处理单个 ContinuousEEG（通常是一个 run）。
    返回: X (N,1,8,500), y_task (N,), y_three (N,)
    """
    x = select_channels(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs)

    kept = filter_left_right_events(eeg.events, eeg.artifacts)

    xs: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []

    for cue, lab_task, lab_three, _ in kept:
        win = task_window_cue_2_to_4(x, int(cue), eeg.fs)
        _append_window(
            xs, y_task, y_three, win, int(lab_task), int(lab_three), eeg.fs
        )

    # 静息：下一 Cue 前 2s（= 该 Cue 前 2s）
    if add_rest and len(kept) > 0:
        starts = extract_rest_cues(
            kept[:, 0],
            eeg.fs,
            x.shape[0],
            rest_sec=2.0,
            task_sec=4.0,  # 上一试次 MI 占用到 cue+4s，避免重叠
        )
        if max_rest is None:
            n_left = int(np.sum(kept[:, 2] == 1))
            n_right = int(np.sum(kept[:, 2] == 2))
            max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        starts = starts[:max_rest]

        for start in starts:
            win = rest_window_with_baseline(
                x, int(start), eeg.fs, win_sec=2.0, baseline_sec=0.5
            )
            _append_window(xs, y_task, y_three, win, 0, 0, eeg.fs)

    if not xs:
        empty_x = np.zeros((0, 1, 8, N_TIMES), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy()

    X = to_model_tensor(xs)
    return (
        X,
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
    )


WIN_SEC_4S = 4.0
N_TIMES_4S = 1000  # 4s @ 250Hz


def preprocess_run_4s(
    eeg: ContinuousEEG,
    add_rest: bool = True,
    max_rest: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    4s 切窗：任务 = Cue 后 0~4s；静息 = 下一 Cue 前 4s → (N,1,8,1000)。
    """
    x = select_channels(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs)

    kept = filter_left_right_events(eeg.events, eeg.artifacts)

    xs: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []

    for cue, lab_task, lab_three, _ in kept:
        win = task_window_cue_0_to_4(x, int(cue), eeg.fs)
        if win is None:
            continue
        win = resample_to_1000(win, fs_in=eeg.fs, fs_out=250.0, win_sec=WIN_SEC_4S)
        if win.shape != (N_TIMES_4S, 8):
            continue
        win = trial_zscore(win)
        xs.append(win)
        y_task.append(int(lab_task))
        y_three.append(int(lab_three))

    if add_rest and len(kept) > 0:
        starts = extract_rest_cues(
            kept[:, 0],
            eeg.fs,
            x.shape[0],
            rest_sec=4.0,
            task_sec=4.0,
        )
        if max_rest is None:
            n_left = int(np.sum(kept[:, 2] == 1))
            n_right = int(np.sum(kept[:, 2] == 2))
            max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        starts = starts[:max_rest]

        for start in starts:
            win = rest_window_with_baseline(
                x, int(start), eeg.fs, win_sec=4.0, baseline_sec=0.5
            )
            if win is None:
                continue
            win = resample_to_1000(win, fs_in=eeg.fs, fs_out=250.0, win_sec=WIN_SEC_4S)
            if win.shape != (N_TIMES_4S, 8):
                continue
            win = trial_zscore(win)
            xs.append(win)
            y_task.append(0)
            y_three.append(0)

    if not xs:
        empty_x = np.zeros((0, 1, 8, N_TIMES_4S), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy()

    X = to_model_tensor(xs)
    return (
        X,
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
    )


def preprocess_run_1s(
    eeg: ContinuousEEG,
    add_rest: bool = True,
    max_rest: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1 s / 40 ms 协议：
      Task = Cue 后 0~4 s 源段 → slide_1s
      Rest = Cue 前 4 s（可缩短避让）→ slide_1s
    返回 X (N,1,8,250), y_task, y_three, trial_id。
    """
    x = select_channels(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs)

    kept = filter_left_right_events(eeg.events, eeg.artifacts)

    xs: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []
    trial_ids: list[int] = []
    tid = 0

    for cue, lab_task, lab_three, _ in kept:
        seg = task_window_cue_0_to_4(x, int(cue), eeg.fs)
        if seg is None:
            continue
        wins = segment_to_1s_windows(seg, eeg.fs)
        if not wins:
            continue
        for w in wins:
            if w.shape != (N_TIMES_1S, 8):
                continue
            xs.append(w)
            y_task.append(int(lab_task))
            y_three.append(int(lab_three))
            trial_ids.append(tid)
        tid += 1

    if add_rest and len(kept) > 0:
        sources = iter_rest_sources_cue_before(
            kept[:, 0], eeg.fs, x.shape[0], rest_sec=4.0, task_sec=4.0
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
            wins = segment_to_1s_windows(seg, eeg.fs)
            for w in wins:
                if w.shape != (N_TIMES_1S, 8):
                    continue
                xs.append(w)
                y_task.append(0)
                y_three.append(0)
                trial_ids.append(tid)
            if wins:
                tid += 1

    if not xs:
        empty_x = np.zeros((0, 1, 8, N_TIMES_1S), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy(), empty_y.copy()

    X = to_model_tensor(xs)
    return (
        X,
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )


def preprocess_run_2s_hop100(
    eeg: ContinuousEEG,
    add_rest: bool = True,
    max_rest: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    旁路 2 s / 100 ms：
      Task = Cue 后 0~4 s → slide
      Rest = Cue 前 4 s（可缩短避让）→ slide
    返回 X (N,1,8,500), y_task, y_three, trial_id。
    """
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


def preprocess_run_3s_hop100(
    eeg: ContinuousEEG,
    add_rest: bool = True,
    max_rest: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    3 s / 100 ms（与 OpenBMI 3s 对齐）：
      Task = Cue 后 0~4 s → slide
      Rest = Cue 前 4 s → slide
    返回 X (N,1,8,750), y_task, y_three, trial_id。
    """
    x = select_channels(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs)

    kept = filter_left_right_events(eeg.events, eeg.artifacts)

    xs: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []
    trial_ids: list[int] = []
    tid = 0
    n_times = N_TIMES_3S

    for cue, lab_task, lab_three, _ in kept:
        seg = task_window_cue_0_to_4(x, int(cue), eeg.fs)
        if seg is None:
            continue
        wins = segment_to_3s_hop100_windows(seg, eeg.fs)
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
            min_win_sec=WIN_SEC_3S_HOP100,
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
            wins = segment_to_3s_hop100_windows(seg, eeg.fs)
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


def preprocess_subject(
    mat_path: Path,
    add_rest: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, yt, y3 = [], [], []
    for eeg in load_bci2a_mat(mat_path):
        X, y_task, y_three = preprocess_run(eeg, add_rest=add_rest)
        if len(y_task):
            Xs.append(X)
            yt.append(y_task)
            y3.append(y_three)
    if not Xs:
        empty_x = np.zeros((0, 1, 8, N_TIMES), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy()
    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(yt, axis=0),
        np.concatenate(y3, axis=0),
    )


def sanity_check_outputs(X, y_task, y_three, n_times: int | None = None) -> None:
    assert len(X) > 0, "没有有效试次"
    t = int(n_times) if n_times is not None else int(X.shape[-1])
    assert X.ndim == 4 and X.shape[1:] == (1, 8, t), X.shape
    assert len(X) == len(y_task) == len(y_three)
    assert set(np.unique(y_task)).issubset({0, 1})
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    assert np.all((y_three == 0) == (y_task == 0))
    assert np.all(y_task[y_three > 0] == 1)
    assert np.isfinite(X).all()
    print(
        "OK",
        "X", X.shape,
        "y_task", np.bincount(y_task, minlength=2),
        "y_three", np.bincount(y_three, minlength=3),
    )


def main() -> None:
    mat_path = Path(r"D:\cyy\MI\DATA\bci2a\A01T.mat")
    out_dir = Path(__file__).resolve().parents[3] / "out" / "bci2a_2s"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_task, y_three = preprocess_subject(mat_path, add_rest=True)
    sanity_check_outputs(X, y_task, y_three)

    np.save(out_dir / "A01_X.npy", X)
    np.save(out_dir / "A01_y_task.npy", y_task)
    np.save(out_dir / "A01_y_three.npy", y_three)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
