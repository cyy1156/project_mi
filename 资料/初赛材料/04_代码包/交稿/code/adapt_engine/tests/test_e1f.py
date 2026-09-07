"""E1f 融合与试次早停单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapt_engine.e1f import apply_temperature_probs, fuse_member_probs  # noqa: E402
from adapt_engine.readout import e1f_conf_stop_from_judgments  # noqa: E402


def test_fuse_member_probs_uniform_temp():
    p0 = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    p1 = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    out = fuse_member_probs([p0, p1], temperatures=[1.0, 1.0], weights=[0.5, 0.5])
    assert out.shape == (3,)
    assert abs(float(out.sum()) - 1.0) < 1e-5


def test_apply_temperature_probs():
    p = np.array([0.9, 0.05, 0.05], dtype=np.float32)
    out = apply_temperature_probs(p, 2.0)
    assert out.argmax() == 0
    assert abs(float(out.sum()) - 1.0) < 1e-5


def test_e1f_conf_stop_picks_first_high_conf():
    js = [
        {"t_rel": 3.0, "p_three": [0.34, 0.33, 0.33], "pred": 0},
        {"t_rel": 3.1, "p_three": [0.2, 0.75, 0.05], "pred": 1},
        {"t_rel": 3.2, "p_three": [0.1, 0.1, 0.8], "pred": 2},
    ]
    out = e1f_conf_stop_from_judgments(js, smooth_radius=0, tau_conf=0.7)
    assert out is not None
    assert out["pred"] == 1
    assert out["rule"] == "e1f_conf_stop"


def test_streaming_S_waits_for_n_plus_1():
    from adapt_engine.readout import streaming_conf_stop_S

    # center0 在 W1 到达后才可评；低置信 → 继续；W2 后 center1 高置信
    probs = [
        np.array([0.40, 0.30, 0.30], np.float32),
        np.array([0.40, 0.30, 0.30], np.float32),
        np.array([0.05, 0.90, 0.05], np.float32),
    ]
    t = [3.0, 3.1, 3.2]
    out = streaming_conf_stop_S(probs, t_rels=t, tau_conf=0.7)
    assert out["pred"] == 1
    assert out["center_idx"] == 1
    assert abs(out["t_dec"] - 3.2) < 1e-6  # n+1 到达时刻


def test_streaming_C_uses_causal_only_and_min3():
    from adapt_engine.readout import streaming_conf_stop_C

    probs = [
        np.array([0.05, 0.90, 0.05], np.float32),  # 高置信但 n=0 不允许提交
        np.array([0.34, 0.33, 0.33], np.float32),
        np.array([0.34, 0.33, 0.33], np.float32),
        np.array([0.05, 0.05, 0.90], np.float32),
    ]
    t = [3.0, 3.1, 3.2, 3.3]
    out = streaming_conf_stop_C(probs, t_rels=t, tau_conf=0.7, min_windows=3)
    # n=2 时 mean(p0,p1,p2) 不足以 ≥0.7；n=3 时 mean(p1,p2,p3) 偏 Right
    assert out["center_idx"] >= 2
    assert out["t_dec"] >= 3.2


def test_streaming_S_matches_offline_bidir_conf_stop_center():
    """齐窗时 S 流式提交中心应与先 r=1 平滑再 conf_stop 一致。"""
    from adapt_engine.readout import (
        _smooth_prob_sequence,
        streaming_conf_stop_S,
    )

    rng = np.random.default_rng(0)
    probs = [rng.dirichlet([1, 1, 1]).astype(np.float32) for _ in range(11)]
    tau = 0.55
    smoothed = _smooth_prob_sequence(probs, 1)
    offline_center = len(probs) - 1
    for i, p in enumerate(smoothed):
        if float(np.max(p)) >= tau:
            offline_center = i
            break
    out = streaming_conf_stop_S(probs, tau_conf=tau)
    assert out["center_idx"] == offline_center


def test_majority_vote_from_probs():
    from adapt_engine.readout import majority_vote_from_probs

    probs = [
        np.array([0.1, 0.8, 0.1], np.float32),
        np.array([0.1, 0.7, 0.2], np.float32),
        np.array([0.2, 0.1, 0.7], np.float32),
    ]
    out = majority_vote_from_probs(probs, t_rels=[3.0, 3.1, 3.2])
    assert out["pred"] == 1
    assert abs(out["t_dec"] - 3.2) < 1e-6
