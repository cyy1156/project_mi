"""滑窗与 1s 预处理冒烟（单元 + 可选单文件）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.steps.slide_1s import (
    N_TIMES_1S,
    n_windows_for_duration,
    segment_to_1s_windows,
    slide_windows_1s,
)


def test_nw_formula() -> None:
    assert n_windows_for_duration(0.9) == 0
    assert n_windows_for_duration(1.0) == 1
    assert n_windows_for_duration(2.0) == 26
    assert n_windows_for_duration(4.0) == 76
    assert n_windows_for_duration(6.0) == 126
    print("n_windows_for_duration OK")


def test_slide_synthetic() -> None:
    x = np.random.randn(1000, 8).astype(np.float64)  # 4s @ 250
    wins = slide_windows_1s(x, fs=250.0)
    assert len(wins) == 76
    assert all(w.shape == (N_TIMES_1S, 8) for w in wins)
    wins2 = segment_to_1s_windows(x, fs_in=250.0)
    assert len(wins2) == 76
    print("slide_synthetic OK")


def test_bci2a_one_file() -> None:
    mat = Path(r"D:/cyy/MI/DATA/bci2a/A01T.mat")
    if not mat.is_file():
        print("skip bci2a real-file (missing)")
        return
    from src.datasets.bci2a.load_mat import load_bci2a_mat
    from src.datasets.bci2a.pipeline import preprocess_run_1s, sanity_check_outputs

    runs = load_bci2a_mat(mat)
    Xs, yts, y3s = [], [], []
    for eeg in runs:
        X, yt, y3, _tid = preprocess_run_1s(eeg, add_rest=True)
        if len(yt):
            Xs.append(X)
            yts.append(yt)
            y3s.append(y3)
    X = np.concatenate(Xs, axis=0)
    yt = np.concatenate(yts, axis=0)
    y3 = np.concatenate(y3s, axis=0)
    sanity_check_outputs(X, yt, y3, n_times=250)
    assert X.shape[-1] == 250
    print("bci2a A01T 1s OK", X.shape)


def test_stieger_one_file() -> None:
    root = Path(r"D:/cyy/MI/DATA/stieger")
    mats = sorted(root.glob("S*_Session_*.mat"))
    if not mats:
        print("skip stieger real-file (missing)")
        return
    # 选较小/靠前的一个做冒烟
    mat = mats[0]
    from src.datasets.stieger.pipeline_1s import preprocess_session_1s, sanity_check_outputs

    X, yt, y3, sid, tid, stats = preprocess_session_1s(mat)
    print("stieger smoke", mat.name, "stats", stats)
    if len(yt) == 0:
        print("WARN: no windows from", mat.name)
        return
    sanity_check_outputs(X, yt, y3)
    assert X.shape[-1] == 250
    print("stieger 1s OK", X.shape)


def main() -> None:
    test_nw_formula()
    test_slide_synthetic()
    test_bci2a_one_file()
    test_stieger_one_file()
    print("ALL smoke OK")


if __name__ == "__main__":
    main()
