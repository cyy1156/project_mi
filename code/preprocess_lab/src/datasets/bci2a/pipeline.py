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
    rest_window_with_baseline,
)
from src.common.steps.resample_zscore import (
    resample_to_1000,
    trial_zscore,
    to_model_tensor,
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


def sanity_check_outputs(X, y_task, y_three) -> None:
    assert len(X) > 0, "没有有效试次"
    assert X.ndim == 4 and X.shape[1:] == (1, 8, N_TIMES), X.shape
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
    mat_path = Path(r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\bci2a\A01T.mat")
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
