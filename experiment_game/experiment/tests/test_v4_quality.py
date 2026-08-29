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


def _openbmi_cfg() -> SignalQualityConfig:
    """OpenBMI 量级回归：8 导全评、旧阈值。"""
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


def _make_cyy_like_window(rng: np.random.Generator | None = None) -> np.ndarray:
    """模拟 cyy 6 导帽：6 导 ~0.7µV std + 大 DC 偏置；CZ/CPZ 卡轨（设备序）。

    2026-08-29 冻结设备序：FC3(0) C3(1) CP3(2) CZ(3) CPZ(4) FC4(5) C4(6) CP4(7)。
    """
    rng = rng or np.random.default_rng(0)
    x = np.zeros((750, 8), dtype=np.float64)
    x[:, 3] = -4190.95263671875  # CZ
    x[:, 4] = -4190.95263671875  # CPZ
    spec = [
        (1, -340.0, 0.65),  # C3
        (6, -48.0, 0.59),   # C4
        (2, -538.0, 0.74),  # CP3
        (5, -384.0, 0.78),  # FC4
        (0, -37.0, 0.75),   # FC3
        (7, -285.0, 0.68),  # CP4
    ]
    for idx, off, std in spec:
        x[:, idx] = off + rng.normal(0, std, 750)
    return x


def test_diagnose_cyy_6ch_cap_passes():
    cfg = V4Config.load_yaml()
    sq = cfg.signal_quality_config()
    d = diagnose_eeg_window(_make_cyy_like_window(), sq, channel_names=list(cfg.channel_labels))
    assert d["window_ok"] is True
    # 大 DC 不进 peak_uv；AC peak 应远小于 max_peak
    assert d["metrics"]["peak_uv"] < 50.0
    assert d["metrics"]["peak_raw_uv"] > 400.0
    cz = next(c for c in d["per_channel"] if c["name"] == "CZ")
    cpz = next(c for c in d["per_channel"] if c["name"] == "CPZ")
    assert cz["reason"] == "unused_expected"
    assert cpz["reason"] == "unused_expected"


def test_large_dc_offset_does_not_saturate():
    """导间数百 µV 直流偏置不应触发 saturation（peak 按 demean 后 AC）。"""
    from experiment_game.experiment.signal_quality import assess_eeg_window

    rng = np.random.default_rng(9)
    x = _make_cyy_like_window(rng)
    # 再加大 DC，模拟 141555 量级（C3@1、CP3@2）
    x[:, 1] -= 800.0
    x[:, 2] -= 600.0
    cfg = V4Config.load_yaml()
    r = assess_eeg_window(x, cfg.signal_quality_config())
    assert r["ok"] is True
    assert r["metrics"]["peak_raw_uv"] > 600.0
    assert r["metrics"]["peak_uv"] < 50.0


def test_diagnose_dead_channel_flags_scoring_channel():
    rng = np.random.default_rng(1)
    x = _make_cyy_like_window(rng)
    # C3@1 死通道：取其余计分导均值（排除 CZ@3/CPZ@4 卡轨）
    x[:, 1] = x[:, [0, 2, 5, 6, 7]].mean(axis=1)
    cfg = V4Config.load_yaml()
    d = diagnose_eeg_window(x, cfg.signal_quality_config(), channel_names=list(cfg.channel_labels))
    assert d["window_ok"] is False
    c3 = next(c for c in d["per_channel"] if c["name"] == "C3")
    assert c3["reason"] == "dead_channel"


def test_diagnose_common_mode_problems():
    """近纯共模：CAR 后几乎无差分 → dead_channel。"""
    rng = np.random.default_rng(2)
    s = rng.normal(0, 40.0, (750, 1))
    x = np.repeat(s, 8, axis=1) + rng.normal(0, 0.3, (750, 8))
    d = diagnose_eeg_window(x, _openbmi_cfg())
    assert d["window_ok"] is False
    assert any(p["reason"] in ("dead_channel", "common_mode") for p in d["problems"])


def test_diagnose_very_flat_fails():
    """极平信号（std≈0.3）应 FAIL。"""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.3, (750, 8))
    cfg = V4Config.load_yaml()
    d = diagnose_eeg_window(x, cfg.signal_quality_config())
    assert d["window_ok"] is False


def test_diagnose_openbmi_like_passes():
    """OpenBMI 采集滤波后典型幅值（med_std≈15–25）应 PASS（8 导旧阈值）。"""
    rng = np.random.default_rng(4)
    x = rng.normal(0, 18.0, (750, 8))
    d = diagnose_eeg_window(x, _openbmi_cfg())
    assert d["window_ok"] is True


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
    hist = [{"window_ok": True, "metrics": {"median_std_uv": 0.7, "common_mode_ratio": 0.5}}] * 5
    s = summarize_v4_session(
        hist,
        duration_s=15.0,
        pass_streak_required=5,
        achieved_stable=True,
        time_to_stable_s=15.0,
        unused_channels=["CZ", "CPZ"],
        scoring_channels=["C3", "C4", "CP3", "FC4", "FC3", "CP4"],
    )
    assert s["verdict"] == "pass"


def _replay_pass_rate(session_name: str) -> tuple[int, int]:
    import pandas as pd
    from experiment_game.experiment.channel_layout import DEVICE_CHANNEL_LABELS

    p = _ROOT / "data" / "sessions" / session_name / "eeg.csv"
    if not p.is_file():
        return 0, 0
    cfg = V4Config.load_yaml()
    sq = cfg.signal_quality_config()
    names = list(cfg.channel_labels)
    df = pd.read_csv(p)
    cols = []
    for name in DEVICE_CHANNEL_LABELS:
        if name in df.columns:
            cols.append(name)
        elif name.upper() in df.columns:
            cols.append(name.upper())
        else:
            alt = next((c for c in df.columns if c.upper() == name.upper()), None)
            if alt is None:
                return 0, 0
            cols.append(alt)
    X = df[cols].values.astype(np.float64)
    win = 750
    oks = total = 0
    for s in range(0, len(X) - win + 1, win):
        d = diagnose_eeg_window(X[s : s + win], sq, channel_names=names)
        total += 1
        if d["window_ok"]:
            oks += 1
    return oks, total


def test_cyy_session_csv_replay_passes():
    """cyy_ws01 真机 CSV 在 6 导 + demean peak 规则下应绝大多数窗 PASS。"""
    oks, total = _replay_pass_rate("cyy_ws01_20260826_132838")
    if total == 0:
        return
    assert total >= 25
    assert oks / total >= 0.85


def test_cyy_141555_high_dc_replay_passes():
    """此前因 RAW peak 报红的会话，demean 后应 PASS。"""
    oks, total = _replay_pass_rate("cyy_ws01_20260826_141555")
    if total == 0:
        return
    assert total >= 10
    assert oks / total >= 0.85
