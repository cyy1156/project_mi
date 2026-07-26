"""Stieger 专用预处理流水线（位于 datasets/stieger/，与 bci2a 分离）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.stieger.load_mat import load_stieger_mat, StiegerTrial
from src.datasets.stieger.labels import map_target
from src.datasets.stieger.paradigm import keep_trial, ALLOWED_TARGETS_BY_TASK
from src.datasets.stieger.windows import extract_mi_or_rest_window
from src.common.steps.select_channels import select_channels, TARGET_CHANNELS
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.resample_zscore import (
    resample_to_1000,
    trial_zscore,
    to_model_tensor,
)

# 与 BCI2a 现行对齐：分类窗 2s → 500@250Hz（取反馈段最后 2s）
WIN_SEC = 2.0
FS_OUT = 250.0
N_TIMES = int(WIN_SEC * FS_OUT)  # 500


def _process_one_trial(tr: StiegerTrial) -> tuple[np.ndarray, int, int] | None:
    if not keep_trial(
        tr.tasknumber,
        tr.targetnumber,
        tr.artifact,
        tr.triallength,
        use_tasks=(1, 2, 3),  # LR + UD + 2D
        min_feedback_sec=WIN_SEC,
    ):
        return None

    labels = map_target(tr.targetnumber)
    if labels is None:
        return None
    y_task, y_three = labels

    # 通道可能因命名大小写不一致：统一 strip
    ch_names = [c.strip() for c in tr.ch_names]
    try:
        x = select_channels(tr.x, ch_names)
    except KeyError:
        # 尝试常见别名（若官方标签带空格等）
        alias = {n.upper(): n for n in ch_names}
        mapped = []
        for want in TARGET_CHANNELS:
            key = want.upper()
            if key not in alias:
                return None
            mapped.append(ch_names.index(alias[key]))
        x = tr.x[:, mapped]

    x = car_reference(x)
    x = notch_and_bandpass(x, tr.fs)

    win = extract_mi_or_rest_window(
        x,
        tr.time_ms,
        tr.fs,
        resultind=tr.resultind,
        feedback_t_ms=2000.0,
        win_sec=WIN_SEC,
        baseline_sec=0.5,
    )
    if win is None or win.shape[0] != int(round(WIN_SEC * tr.fs)):
        return None

    win = resample_to_1000(win, fs_in=tr.fs, fs_out=FS_OUT, win_sec=WIN_SEC)
    if win.shape != (N_TIMES, 8):
        return None
    win = trial_zscore(win)
    return win, int(y_task), int(y_three)


def preprocess_session(
    mat_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    单会话 → X, y_task, y_three, subjects, stats
    """
    mat_path = Path(mat_path)
    trials = load_stieger_mat(mat_path)
    xs, yt, y3, sids = [], [], [], []
    stats = {
        "n_raw": len(trials),
        "n_keep": 0,
        "n_drop_short": 0,
        "n_drop_artifact": 0,
        "n_drop_target": 0,
        "n_drop_task": 0,
    }

    for tr in trials:
        # 统计用：与 keep_trial 同一套规则（含 task 2/3）
        if int(tr.tasknumber) not in (1, 2, 3):
            stats["n_drop_task"] += 1
            continue
        if int(tr.artifact) == 1:
            stats["n_drop_artifact"] += 1
            continue
        # 该 task 不允许的 target（含双手 up、UD 的左右等）
        allowed = ALLOWED_TARGETS_BY_TASK.get(int(tr.tasknumber), set())
        if int(tr.targetnumber) not in allowed or map_target(tr.targetnumber) is None:
            stats["n_drop_target"] += 1
            continue
        if float(tr.triallength) < WIN_SEC:
            stats["n_drop_short"] += 1
            continue

        out = _process_one_trial(tr)
        if out is None:
            stats["n_drop_short"] += 1
            continue
        win, y_task, y_three = out
        xs.append(win)
        yt.append(y_task)
        y3.append(y_three)
        sids.append(tr.subject)

    stats["n_keep"] = len(xs)
    if not xs:
        empty = np.zeros((0, 1, 8, N_TIMES), np.float32)
        z = np.zeros((0,), np.int64)
        return empty, z, z.copy(), np.array([], dtype=object), stats

    X = to_model_tensor(xs)
    return (
        X,
        np.asarray(yt, dtype=np.int64),
        np.asarray(y3, dtype=np.int64),
        np.asarray(sids, dtype=object),
        stats,
    )


def sanity_check_outputs(X, y_task, y_three) -> None:
    assert len(X) > 0, "没有有效试次"
    assert X.ndim == 4 and X.shape[1:] == (1, 8, N_TIMES)
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
    # 调试：先跑一个会话
    mat_path = Path(
        r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\stieger\S1_Session_10.mat"
    )
    out_dir = Path(__file__).resolve().parents[3] / "out" / "stieger_2s"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_task, y_three, subjects, stats = preprocess_session(mat_path)
    print("stats:", stats)
    sanity_check_outputs(X, y_task, y_three)

    np.save(out_dir / "debug_S1S1_X.npy", X)
    np.save(out_dir / "debug_S1S1_y_task.npy", y_task)
    np.save(out_dir / "debug_S1S1_y_three.npy", y_three)
    np.save(out_dir / "debug_S1S1_subjects.npy", subjects)
    print("saved to", out_dir, "X.shape", X.shape)


if __name__ == "__main__":
    main()