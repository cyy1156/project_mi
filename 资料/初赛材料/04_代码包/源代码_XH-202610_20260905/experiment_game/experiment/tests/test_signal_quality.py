"""信号质量门控单测。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiment_game.experiment.signal_quality import (  # noqa: E402
    SignalQualityConfig,
    assess_eeg_window,
    summarize_baseline_hat_check,
)


def _openbmi_cfg() -> SignalQualityConfig:
    return SignalQualityConfig(
        min_median_std_uv=5.0,
        min_peak_to_peak_uv=40.0,
        max_peak_uv=800.0,
        min_per_channel_std_uv=3.0,
        min_active_channels=6,
        max_channel_std_ratio=25.0,
        max_median_std_uv=120.0,
        max_ptp_uv=800.0,
        min_car_std_uv=1.2,
        max_common_mode_ratio=1.25,
    )


def test_flatline_rejected():
    x = np.zeros((750, 8))
    r = assess_eeg_window(x)
    assert r["ok"] is False
    assert r["reason"] in ("flatline", "low_variance", "low_dynamics")


def test_normal_eeg_ok():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 25.0, (750, 8))
    r = assess_eeg_window(x, _openbmi_cfg())
    assert r["ok"] is True


def test_saturation_rejected():
    x = np.random.default_rng(1).normal(0, 20.0, (750, 8))
    x[100, 0] = 5000.0
    r = assess_eeg_window(x, SignalQualityConfig(max_peak_uv=1000.0))
    assert r["ok"] is False
    assert r["reason"] == "saturation"


def test_channel_imbalance_rejected():
    x = np.random.default_rng(2).normal(0, 5.0, (750, 8))
    x[:, 0] *= 30.0
    r = assess_eeg_window(
        x,
        SignalQualityConfig(max_channel_std_ratio=20.0, max_peak_uv=5000.0),
    )
    assert r["ok"] is False
    assert r["reason"] == "channel_imbalance"


def test_artifact_rejected():
    x = np.random.default_rng(3).normal(0, 150.0, (750, 8))
    r = assess_eeg_window(x, _openbmi_cfg())
    assert r["ok"] is False
    assert r["reason"] == "artifact"


def test_dead_channel_rejected():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 15.0, (750, 8))
    x[:, 2] = x[:, [0, 1, 3, 4, 5, 6, 7]].mean(axis=1)
    r = assess_eeg_window(x, _openbmi_cfg())
    assert r["ok"] is False
    assert r["reason"] == "dead_channel"
    assert r["metrics"]["dead_channel_idx"] == 2


def test_common_mode_rejected():
    rng = np.random.default_rng(5)
    s = rng.normal(0, 40.0, (750, 1))
    x = np.repeat(s, 8, axis=1) + rng.normal(0, 0.5, (750, 8))
    r = assess_eeg_window(x, _openbmi_cfg())
    assert r["ok"] is False
    assert r["reason"] in ("common_mode", "dead_channel")


def test_baseline_hat_pass():
    rng = np.random.default_rng(6)
    baseline = rng.normal(0, 12.0, (7500, 8))
    hat = summarize_baseline_hat_check(baseline, fs=250.0, cfg=_openbmi_cfg())
    assert hat["verdict"] == "pass"
    assert hat["n_windows"] == 10


def test_baseline_hat_fail_on_sync():
    rng = np.random.default_rng(7)
    s = rng.normal(0, 50.0, (7500, 1))
    baseline = np.repeat(s, 8, axis=1)
    hat = summarize_baseline_hat_check(baseline, fs=250.0, cfg=_openbmi_cfg())
    assert hat["verdict"] == "fail"
    assert hat["bad_frac"] >= 0.5
