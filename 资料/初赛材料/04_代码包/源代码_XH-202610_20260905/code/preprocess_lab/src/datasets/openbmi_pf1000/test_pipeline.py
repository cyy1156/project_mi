"""单元测试：不依赖真实 mat。"""
from __future__ import annotations

import numpy as np

from src.common.steps.resample_zscore import to_model_tensor
from src.datasets.openbmi_pf1000.pipeline import (
    CUT_SPEC,
    FUT,
    N_TIMES,
    SEG_SEC,
    FS_OUT,
    iter_rest_origins_pf1000,
    windows_from_seg250,
    sanity_check_pf1000,
)


def test_windows_geometry():
    rng = np.random.default_rng(0)
    seg = rng.normal(size=(1400, 8)).astype(np.float32)
    wins = windows_from_seg250(seg, lab_task=1, lab_three=1, zscore=True)
    # t0=0.4..2.0 hop0.1 → 17
    assert len(wins) == 17, len(wins)
    x, yt, y3, t0 = wins[0]
    assert x.shape == (1000, 8)
    assert abs(t0 - 0.4) < 1e-6
    assert yt == 1 and y3 == 1
    # 缺 past 的 i0=0 不应出现
    assert all(w[3] >= 0.4 - 1e-6 for w in wins)


def test_rest_windows_same_geometry():
    rng = np.random.default_rng(2)
    seg = rng.normal(size=(int(SEG_SEC * FS_OUT), 8)).astype(np.float32)
    wins = windows_from_seg250(seg, lab_task=0, lab_three=0, zscore=False)
    assert len(wins) == 17
    assert all(w[1] == 0 and w[2] == 0 for w in wins)
    assert CUT_SPEC["label_map"]["rest"] == 0


def test_rest_origins_no_overlap_shorten():
    fs = 1000.0
    # cue0=0, cue1=10000 → origin1=10000-5600=4400；prev_task_end=4000 → ok
    cues = np.array([0, 10000, 20000], dtype=int)
    origins = iter_rest_origins_pf1000(cues, fs, n_times=30000)
    assert origins == [4400, 14400]
    # cue 太近：prev_task 侵占 → 丢弃
    cues2 = np.array([0, 5000], dtype=int)  # origin=5000-5600=-600 → skip; also would overlap
    assert iter_rest_origins_pf1000(cues2, fs, n_times=10000) == []
    cues3 = np.array([0, 8000], dtype=int)  # origin=2400, prev_end=4000 → overlap drop
    assert iter_rest_origins_pf1000(cues3, fs, n_times=10000) == []


def test_mask_tail_zero_three_class():
    rng = np.random.default_rng(1)
    seg = rng.normal(size=(1400, 8)).astype(np.float32)
    wins = windows_from_seg250(seg, lab_task=0, lab_three=0, zscore=False)
    xs = [w[0] for w in wins]
    X_full = to_model_tensor(xs)
    X_mask = X_full.copy()
    X_mask[..., -FUT:] = 0
    y3 = np.zeros((len(xs),), dtype=np.int64)
    yt = np.zeros((len(xs),), dtype=np.int64)
    sanity_check_pf1000(X_full, X_mask, y3, y_task=yt)
    assert X_full.shape[-1] == N_TIMES


def test_short_seg_no_windows():
    seg = np.zeros((1000, 8), np.float32)  # <1400
    wins = windows_from_seg250(seg, lab_task=1, lab_three=1, zscore=False)
    assert wins == []


if __name__ == "__main__":
    test_windows_geometry()
    test_rest_windows_same_geometry()
    test_rest_origins_no_overlap_shorten()
    test_mask_tail_zero_three_class()
    test_short_seg_no_windows()
    print("OK")
