from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eeg_types import ContinuousEEG
from src.io.load_bci2a_mat import load_bci2a_mat
from src.steps.select_channels import select_channels
from src.steps.filter_car import car_reference, notch_and_bandpass
from src.steps.harmonize_labels import (
    filter_left_right_events,
    extract_rest_cues,
)
from src.steps.epoch_baseline import (
    slice_epoch,
    baseline_correct,
    classification_window,
    rest_window_with_baseline,
)
from src.steps.resample_zscore import (
    resample_to_1000,
    trial_zscore,
    to_model_tensor,
)
from src.steps.split_subjects import split_all_trials


def _append_window(
    xs: list,
    y_task: list,
    y_three: list,
    win: np.ndarray | None,
    lab_task: int,
    lab_three: int,
    fs: float,
) -> None:
    """单个 4s 窗：重采样 → z-score → 收入列表。"""
    if win is None:
        return
    win = resample_to_1000(win, fs_in=fs, fs_out=250.0)
    if win.shape != (1000, 8):
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
    返回: X (N,1,8,1000), y_task (N,), y_three (N,)
    """
    x = select_channels(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs)

    kept = filter_left_right_events(eeg.events, eeg.artifacts)
    # kept 每行: [cue, label_task, label_three, trial_index]

    xs: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []

    for cue, lab_task, lab_three, _ in kept:
        ep = slice_epoch(x, int(cue), eeg.fs)
        if ep is None:
            continue
        ep = baseline_correct(ep, eeg.fs)
        win = classification_window(ep, eeg.fs)
        _append_window(
            xs, y_task, y_three, win, int(lab_task), int(lab_three), eeg.fs
        )

    # ----- 静息：下一 Cue 前 4s，标签 (0, 0) -----
    if add_rest and len(kept) > 0:
        starts = extract_rest_cues(kept[:, 0], eeg.fs, x.shape[0])
        if max_rest is None:
            n_left = int(np.sum(kept[:, 2] == 1))
            n_right = int(np.sum(kept[:, 2] == 2))
            max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        starts = starts[:max_rest]

        for start in starts:
            win = rest_window_with_baseline(x, int(start), eeg.fs)
            _append_window(xs, y_task, y_three, win, 0, 0, eeg.fs)

    if not xs:
        empty_x = np.zeros((0, 1, 8, 1000), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy()

    X = to_model_tensor(xs)  # (N, 1, 8, 1000)
    return (
        X,
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
    )


def preprocess_subject(
    mat_path: Path,
    add_rest: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一个 .mat（一名受试者多个 run）→ 合并为 X, y_task, y_three。"""
    Xs, yt, y3 = [], [], []
    for eeg in load_bci2a_mat(mat_path):
        X, y_task, y_three = preprocess_run(eeg, add_rest=add_rest)
        if len(y_task):
            Xs.append(X)
            yt.append(y_task)
            y3.append(y_three)
    if not Xs:
        empty_x = np.zeros((0, 1, 8, 1000), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy()
    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(yt, axis=0),
        np.concatenate(y3, axis=0),
    )


def sanity_check_outputs(X, y_task, y_three) -> None:
    assert len(X) > 0, "没有有效试次"
    assert X.ndim == 4 and X.shape[1:] == (1, 8, 1000)
    assert len(X) == len(y_task) == len(y_three)
    assert set(np.unique(y_task)).issubset({0, 1})
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    assert np.all((y_three == 0) == (y_task == 0))
    assert np.all(y_task[y_three > 0] == 1)
    assert np.isfinite(X).all()
    for i in np.random.choice(len(X), size=min(5, len(X)), replace=False):
        trial = X[i, 0]  # (8, 1000)
        assert np.allclose(trial.mean(axis=1), 0, atol=1e-5)
        assert np.allclose(trial.std(axis=1), 1, atol=1e-4)
    print(
        "OK",
        "X", X.shape,
        "y_task", np.bincount(y_task, minlength=2),
        "y_three", np.bincount(y_three, minlength=3),
    )


def main() -> None:
    mat_path = Path(r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\bci2a\A01T.mat")
    out_dir = Path(__file__).resolve().parents[1] / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 单受试者完整预处理
    X, y_task, y_three = preprocess_subject(mat_path, add_rest=True)
    sanity_check_outputs(X, y_task, y_three)

    # 2) 保存全量（未划分）三件套
    np.save(out_dir / "A01_X.npy", X)
    np.save(out_dir / "A01_y_task.npy", y_task)
    np.save(out_dir / "A01_y_three.npy", y_three)
    print("saved full set to", out_dir)

    # 3) 全体试次 8:2 划分
    subjects = np.array(["A01"] * len(X))
    parts = split_all_trials(
        X, y_task, y_three, val_ratio=0.2, seed=42, subjects=subjects
    )
    X_tr, yt_tr, y3_tr, sid_tr = parts["train"]
    X_va, yt_va, y3_va, sid_va = parts["val"]
    print("train", X_tr.shape, "val", X_va.shape)

    # 4) 保存训练集 / 验证集
    np.save(out_dir / "train_X.npy", X_tr)
    np.save(out_dir / "train_y_task.npy", yt_tr)
    np.save(out_dir / "train_y_three.npy", y3_tr)
    np.save(out_dir / "train_subjects.npy", sid_tr)

    np.save(out_dir / "val_X.npy", X_va)
    np.save(out_dir / "val_y_task.npy", yt_va)
    np.save(out_dir / "val_y_three.npy", y3_va)
    np.save(out_dir / "val_subjects.npy", sid_va)
    print("saved train/val to", out_dir)


if __name__ == "__main__":
    main()
