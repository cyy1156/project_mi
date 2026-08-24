"""v4 信号质量诊断与监控单测。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiment_game.experiment.signal_quality import (  # noqa: E402
    SignalQualityConfig,
    diagnose_eeg_window,
    summarize_v4_session,
)
from experiment_game.experiment.v4_quality import V4QualityMonitor  # noqa: E402
from experiment_game.experiment.v4_config import V4Config  # noqa: E402


def test_diagnose_good_window():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 12.0, (750, 8))
    d = diagnose_eeg_window(x)
    assert d["window_ok"] is True
    assert all(d["per_channel_ok"])


def test_diagnose_dead_channel_flags_channel():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 15.0, (750, 8))
    x[:, 1] = x[:, [0, 2, 3, 4, 5, 6, 7]].mean(axis=1)
    d = diagnose_eeg_window(x)
    assert d["window_ok"] is False
    assert d["per_channel"][1]["reason"] == "dead_channel"


def test_diagnose_common_mode_problems():
    rng = np.random.default_rng(2)
    s = rng.normal(0, 40.0, (750, 1))
    x = np.repeat(s, 8, axis=1) + rng.normal(0, 5.0, (750, 8))
    d = diagnose_eeg_window(x)
    assert d["window_ok"] is False
    assert any(p["reason"] == "common_mode" for p in d["problems"])


def test_v4_monitor_streak_pass():
    cfg = V4Config(pass_streak_required=3)
    mon = V4QualityMonitor(cfg)
    evt = None
    for _ in range(3):
        evt = mon.update({"window_ok": True, "metrics": {}, "per_channel": []}, elapsed_s=1.0)
    assert mon.achieved_stable is True
    assert evt is not None
    assert evt["streak"] == 3


def test_v4_monitor_streak_reset_on_fail():
    cfg = V4Config(pass_streak_required=5)
    mon = V4QualityMonitor(cfg)
    for _ in range(3):
        mon.update({"window_ok": True, "metrics": {}, "per_channel": []}, elapsed_s=1.0)
    mon.update({"window_ok": False, "metrics": {}, "per_channel": []}, elapsed_s=4.0)
    assert mon.streak == 0
    assert mon.achieved_stable is False


def test_summarize_v4_session_pass():
    hist = [{"window_ok": True, "metrics": {"median_std_uv": 10.0, "common_mode_ratio": 0.2}}] * 5
    s = summarize_v4_session(
        hist,
        duration_s=15.0,
        pass_streak_required=5,
        achieved_stable=True,
        time_to_stable_s=15.0,
    )
    assert s["verdict"] == "pass"
