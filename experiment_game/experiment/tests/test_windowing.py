"""core.windowing 单测：常量、形状、与历史 3s/hop100 协议对齐。"""

from __future__ import annotations

import numpy as np

from experiment_game.core.channel_layout import DEVICE_CHANNEL_LABELS
from experiment_game.core.windowing import (
    FROZEN,
    FS,
    HOP_SEC,
    N_TIMES,
    WIN_SEC,
    WINDOWING_VERSION,
    extract_segment_baseline,
    n_windows_3s_hop100,
    segment_to_3s_hop100_windows,
    slide_3s_from_interval,
    task_window_cue_0_to_4,
    wins_to_model,
)


def test_constants_match_openbmi_protocol():
    assert WINDOWING_VERSION == "openbmi_align_v1"
    assert FS == 250.0
    assert WIN_SEC == 3.0
    assert HOP_SEC == 0.1
    assert N_TIMES == 750
    assert FROZEN == list(DEVICE_CHANNEL_LABELS)


def test_n_windows_4s_task_segment():
    # Cue+0~4s → 理论 11 窗（(4-3)/0.1 + 1）
    assert n_windows_3s_hop100(4.0) == 11
    assert n_windows_3s_hop100(3.0) == 1
    assert n_windows_3s_hop100(2.9) == 0


def test_task_window_and_slide_shapes():
    fs = 250.0
    n_ch = 8
    cue = 200
    x = np.zeros((cue + int(5 * fs), n_ch), dtype=np.float64)
    x[cue - 125 : cue, :] = 1.0
    x[cue : cue + int(4 * fs), :] = 3.0
    seg = task_window_cue_0_to_4(x, cue, fs)
    assert seg is not None
    assert seg.shape == (1000, 8)
    assert np.allclose(seg.mean(), 2.0, atol=1e-6)

    wins = segment_to_3s_hop100_windows(seg, fs, zscore=False)
    assert len(wins) == 11
    assert all(w.shape == (750, 8) for w in wins)

    model = wins_to_model(wins)
    assert len(model) == 11
    assert model[0].shape == (8, 750)
    assert model[0].dtype == np.float32


def test_extract_segment_baseline_prefers_pre_t0():
    fs = 250.0
    x = np.zeros((2000, 8))
    x[100:225, :] = 5.0
    x[225:725, :] = 15.0
    seg = extract_segment_baseline(x, 225, 725, fs, baseline_sec=0.5)
    assert seg is not None
    assert seg.shape[0] == 500
    assert np.allclose(seg.mean(), 10.0, atol=1e-5)


def test_slide_3s_from_interval():
    fs = 250.0
    n = int(6 * fs)
    t_lsl = np.arange(n, dtype=np.float64) / fs
    x = np.random.randn(n, 8).astype(np.float64)
    out = slide_3s_from_interval(x, t_lsl, 0.0, 4.0, t0_min=0.0, zscore=True)
    assert len(out) == 11
    assert out[0].shape == (8, 750)


def test_to_nchw_rejects_time_major_windows():
    from experiment_game.core.windowing import to_nchw

    bad = [np.zeros((750, 8), dtype=np.float32)]
    try:
        to_nchw(bad)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "wins_to_model" in str(exc) or "(T,C)" in str(exc)

    good = [np.zeros((8, 750), dtype=np.float32)]
    out = to_nchw(good)
    assert out.shape == (1, 1, 8, 750)


def test_openbmi_align_cut_reexports():
    from experiment_game.offline import openbmi_align_cut as oac

    assert oac.WINDOWING_VERSION == WINDOWING_VERSION
    assert oac.FROZEN == FROZEN
    assert oac.n_windows_3s_hop100(4.0) == 11
