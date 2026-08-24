#!/usr/bin/env python3
"""离线回放 v4 帽检：对 sessions/<id>/eeg.csv 或任意 8 列 CSV 逐窗评估。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from experiment_game.experiment.signal_quality import diagnose_eeg_window, summarize_v4_session  # noqa: E402
from experiment_game.experiment.channel_layout import (  # noqa: E402
    DEVICE_CHANNEL_LABELS,
    reorder_device_to_frozen,
)
from experiment_game.experiment.v4_config import V4Config  # noqa: E402
from experiment_game.experiment.v4_quality import V4QualityMonitor  # noqa: E402


def load_eeg_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        skip = {"lsl_time", "timestamp", "marker"}
        cols = [c for c in reader.fieldnames or [] if c not in skip][:8]
        if len(cols) < 8:
            raise ValueError(f"需要 8 通道列，得到 {cols}")
        for row in reader:
            rows.append([float(row[c]) for c in cols])
    if not rows:
        raise ValueError("CSV 无数据")
    x = np.asarray(rows, dtype=np.float64)
    # eeg.csv 列为设备序（C3,C4,CZ,…），诊断用冻结序
    if cols == DEVICE_CHANNEL_LABELS[:8] or cols == [c.upper() for c in DEVICE_CHANNEL_LABELS]:
        x = reorder_device_to_frozen(x)
    return x


def replay(path: Path, *, cfg: V4Config) -> dict:
    x = load_eeg_csv(path)
    fs = cfg.fs
    win_n = max(1, int(round(cfg.eval_window_s * fs)))
    hop_n = max(1, int(round(cfg.eval_interval_s * fs)))
    sq = cfg.signal_quality_config()
    names = list(cfg.channel_labels)
    mon = V4QualityMonitor(cfg)
    history = []

    for start in range(0, x.shape[0] - win_n + 1, hop_n):
        win = x[start : start + win_n]
        diag = diagnose_eeg_window(win, sq, channel_names=names)
        elapsed = start / fs
        mon.update(diag, elapsed_s=elapsed)
        history.append({**diag, "pass_streak": mon.streak, "rolling_verdict": mon.rolling_verdict()})

    duration = x.shape[0] / fs
    summary = summarize_v4_session(
        history,
        duration_s=duration,
        pass_streak_required=cfg.pass_streak_required,
        achieved_stable=mon.achieved_stable,
        time_to_stable_s=mon.time_to_stable_s,
        channel_names=names,
    )
    return {"summary": summary, "windows": history}


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay v4 hat check on eeg.csv")
    ap.add_argument("csv", type=Path, help="eeg.csv 或 8 列脑电 CSV")
    ap.add_argument("-o", "--out", type=Path, help="输出 v4_report.json")
    ap.add_argument("--config", type=Path, default=None, help="v4_session.yaml")
    args = ap.parse_args()

    cfg = V4Config.load_yaml(args.config) if args.config else V4Config.load_yaml()
    result = replay(args.csv.resolve(), cfg=cfg)
    text = json.dumps(result["summary"], ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
