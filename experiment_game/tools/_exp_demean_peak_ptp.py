"""实验：peak/ptp 前逐通道减均值，在已采集 cyy 会话上对比 PASS 率。

不改正式代码，仅离线回放分析。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_ROOT))

from experiment_game.core.channel_layout import reorder_device_to_frozen
from experiment_game.experiment.signal_quality import SignalQualityConfig, assess_eeg_window
from experiment_game.experiment.v4_config import V4Config

WIN = 750
SESSIONS = [
    _ROOT / "data" / "sessions" / "cyy_ws01_20260826_132838",
    _ROOT / "data" / "sessions" / "cyy_ws01_20260826_141555",
    _ROOT / "data" / "sessions" / "cyy_ws01_20260826_133124",
]


def _load_x(ses: Path):
    for p in (ses / "eeg.csv", ses / "continuous" / "eeg.csv"):
        if p.is_file():
            df = pd.read_csv(p)
            cols = [c for c in df.columns if c != "lsl_time"]
            X = reorder_device_to_frozen(df[cols].values.astype(np.float64))
            return X, p
    return None, None


def _pct(a) -> str:
    a = np.asarray(a, float)
    if a.size == 0:
        return "n/a"
    p5, p50, p95 = np.percentile(a, [5, 50, 95])
    return f"P5={p5:.2f} P50={p50:.2f} P95={p95:.2f} max={a.max():.2f}"


def _raw_peak_ptp(x: np.ndarray, idx: list[int]):
    xs = x[:, idx]
    med = float(np.median(np.std(xs, axis=0)))
    peak = float(np.max(np.abs(xs)))
    ptp = float(np.max(xs) - np.min(xs))
    return med, peak, ptp


def _demeaned_peak_ptp(x: np.ndarray, idx: list[int]):
    xs = x[:, idx].astype(np.float64)
    xs = xs - xs.mean(axis=0, keepdims=True)
    med = float(np.median(np.std(xs, axis=0)))
    peak = float(np.max(np.abs(xs)))
    ptp_global = float(np.max(xs) - np.min(xs))
    ptp_per_ch = float(np.max(np.ptp(xs, axis=0)))
    return med, peak, ptp_global, ptp_per_ch


def _loose_other_gates(sq: SignalQualityConfig) -> SignalQualityConfig:
    """放宽 peak/ptp，只保留其余门控，便于单独评估 demean 后的 peak/ptp。"""
    return SignalQualityConfig(
        enabled=True,
        min_median_std_uv=sq.min_median_std_uv,
        min_peak_to_peak_uv=0.0,
        max_peak_uv=1e9,
        min_per_channel_std_uv=sq.min_per_channel_std_uv,
        min_active_channels=sq.min_active_channels,
        max_channel_std_ratio=sq.max_channel_std_ratio,
        max_median_std_uv=sq.max_median_std_uv,
        max_ptp_uv=1e9,
        min_car_std_uv=sq.min_car_std_uv,
        max_common_mode_ratio=sq.max_common_mode_ratio,
        max_per_channel_std_uv=sq.max_per_channel_std_uv,
        scoring_channel_indices=sq.scoring_channel_indices,
        unused_channel_indices=sq.unused_channel_indices,
        unused_allow_rail=sq.unused_allow_rail,
    )


def main() -> None:
    cfg = V4Config.load_yaml()
    sq = cfg.signal_quality_config()
    scoring = list(cfg.scoring_indices())
    names = [cfg.channel_labels[i] for i in scoring]
    print("scoring:", scoring, names)
    print(
        f"thresholds: min_ptp={sq.min_peak_to_peak_uv} "
        f"max_ptp={sq.max_ptp_uv} max_peak={sq.max_peak_uv}"
    )
    loose = _loose_other_gates(sq)

    for ses in SESSIONS:
        X, path = _load_x(ses)
        if X is None:
            print(f"\nSKIP {ses.name}")
            continue

        raw_peaks, raw_ptps = [], []
        dm_peaks, dm_ptp_g, dm_ptp_c = [], [], []
        meds = []
        reasons_raw: dict[str, int] = {}
        reasons_dm_same: dict[str, int] = {}
        reasons_dm_recal: dict[str, int] = {}
        ok_raw = ok_dm_same = ok_dm_recal = 0
        n = 0

        for s in range(0, len(X) - WIN + 1, WIN):
            win = X[s : s + WIN]
            n += 1
            med, pk, ptp = _raw_peak_ptp(win, scoring)
            _, pk2, ptp_g, ptp_c = _demeaned_peak_ptp(win, scoring)
            meds.append(med)
            raw_peaks.append(pk)
            raw_ptps.append(ptp)
            dm_peaks.append(pk2)
            dm_ptp_g.append(ptp_g)
            dm_ptp_c.append(ptp_c)

            qa = assess_eeg_window(win, sq)
            if qa["ok"]:
                ok_raw += 1
            else:
                r = qa["reason"] or "?"
                reasons_raw[r] = reasons_raw.get(r, 0) + 1

            # demean + 现行阈值（含 min_ptp=480）
            base = assess_eeg_window(win, loose)
            fail = None if base["ok"] else (base["reason"] or "?")
            if fail is None:
                if ptp_g < sq.min_peak_to_peak_uv:
                    fail = "low_dynamics"
                elif pk2 > sq.max_peak_uv:
                    fail = "saturation"
                elif ptp_g > sq.max_ptp_uv:
                    fail = "high_ptp"
            if fail is None:
                ok_dm_same += 1
            else:
                reasons_dm_same[fail] = reasons_dm_same.get(fail, 0) + 1

            # demean + 去掉 min_ptp（AC 峰峰值远小于含 DC 的标定值）
            # 上限仍用 max_peak/max_ptp=600，按 per-channel max ptp
            fail2 = None if base["ok"] else (base["reason"] or "?")
            if fail2 is None:
                if pk2 > sq.max_peak_uv:
                    fail2 = "saturation"
                elif ptp_c > sq.max_ptp_uv:
                    fail2 = "high_ptp"
            if fail2 is None:
                ok_dm_recal += 1
            else:
                reasons_dm_recal[fail2] = reasons_dm_recal.get(fail2, 0) + 1

        print(f"\n==== {ses.name}  n={n}  file={path}")
        print(f"  med_std          {_pct(meds)}")
        print(f"  RAW peak         {_pct(raw_peaks)}")
        print(f"  RAW ptp          {_pct(raw_ptps)}")
        print(f"  DEMEAN peak      {_pct(dm_peaks)}")
        print(f"  DEMEAN ptp_glob  {_pct(dm_ptp_g)}")
        print(f"  DEMEAN ptp_chmax {_pct(dm_ptp_c)}")
        print(
            f"  PASS raw (current):              "
            f"{ok_raw}/{n} = {100 * ok_raw / n:.1f}%  fails={reasons_raw}"
        )
        print(
            f"  PASS demean + SAME thr:          "
            f"{ok_dm_same}/{n} = {100 * ok_dm_same / n:.1f}%  fails={reasons_dm_same}"
        )
        print(
            f"  PASS demean + no min_ptp + max on per-ch ptp: "
            f"{ok_dm_recal}/{n} = {100 * ok_dm_recal / n:.1f}%  fails={reasons_dm_recal}"
        )


if __name__ == "__main__":
    main()
