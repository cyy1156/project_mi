#!/usr/bin/env python3
"""校验 Phase 1 会话：MI/Rest 窗长 ≈4s，events 落在 eeg 时间轴内。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_events(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eeg_time_span(csv_path: Path) -> tuple[float, float, int]:
    import csv

    times: list[float] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "lsl_time" not in reader.fieldnames:
            raise ValueError(f"eeg.csv 缺少 lsl_time 列: {reader.fieldnames}")
        for row in reader:
            times.append(float(row["lsl_time"]))
    if not times:
        raise ValueError("eeg.csv 无数据行")
    return times[0], times[-1], len(times)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, required=True, help="会话目录")
    p.add_argument("--tol", type=float, default=0.05, help="窗长容差（秒）")
    p.add_argument("--min-trials", type=int, default=20, help="最少试次数（Phase1 验收默认 20）")
    args = p.parse_args(argv)

    root = args.session
    events_path = root / "events.jsonl"
    eeg_path = root / "eeg.csv"
    if not events_path.is_file() or not eeg_path.is_file():
        print(f"缺少 events.jsonl 或 eeg.csv: {root}", file=sys.stderr)
        return 2

    events = load_events(events_path)
    t0, t1, n_samp = eeg_time_span(eeg_path)
    print(f"EEG samples={n_samp} lsl_span=[{t0:.3f}, {t1:.3f}] ({t1 - t0:.1f}s)")

    by_trial: dict[int, dict[str, float]] = defaultdict(dict)
    prev_t = None
    regressions = 0
    for ev in events:
        t = float(ev["t_lsl"])
        if prev_t is not None and t + 1e-9 < prev_t:
            regressions += 1
        prev_t = t
        tid = ev.get("trial_id")
        if tid is None:
            continue
        name = ev["event"]
        if name in ("mi_start", "mi_end", "rest_start", "rest_end", "cue"):
            by_trial[int(tid)][name] = t

    mi_ok = rest_ok = cue_in_span = 0
    mi_bad = rest_bad = []
    for tid, m in sorted(by_trial.items()):
        if "mi_start" in m and "mi_end" in m:
            dur = m["mi_end"] - m["mi_start"]
            if abs(dur - 4.0) <= args.tol:
                mi_ok += 1
            else:
                mi_bad.append((tid, dur))
        if "rest_start" in m and "rest_end" in m:
            dur = m["rest_end"] - m["rest_start"]
            if abs(dur - 4.0) <= args.tol:
                rest_ok += 1
            else:
                rest_bad.append((tid, dur))
        if "cue" in m and t0 <= m["cue"] <= t1:
            cue_in_span += 1

    n_trials = len(by_trial)
    print(f"trials with stage marks: {n_trials}")
    print(f"MI 4s ok: {mi_ok}/{n_trials}  bad={mi_bad[:5]}")
    print(f"Rest 4s ok: {rest_ok}/{n_trials}  bad={rest_bad[:5]}")
    print(f"cue within EEG span: {cue_in_span}/{n_trials}")
    print(f"t_lsl regressions: {regressions}")

    passed = (
        n_trials >= 1
        and mi_ok == n_trials
        and rest_ok == n_trials
        and cue_in_span == n_trials
        and regressions == 0
    )
    if n_trials >= args.min_trials:
        print(f"trial count ≥{args.min_trials}: OK")
    else:
        print(f"trial count ≥{args.min_trials}: FAIL ({n_trials})")
        passed = False

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
