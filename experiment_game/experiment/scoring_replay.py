"""MI 多数票离线回放（与在线 trial_v2 / MiTrialTracker 同规则）。

用法：
  python -m experiment_game.experiment.scoring_replay path/to/events.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from experiment_game.experiment.trial_scoring import MiTrialTracker


def load_judge_rows(events_path: Path) -> Dict[int, List[dict]]:
    """trial_id → 按 t_rel 排序的 judge 行。"""
    by_trial: Dict[int, List[dict]] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "judge":
            continue
        if row.get("signal_bad"):
            by_trial.setdefault(int(row["trial_id"]), []).append({"signal_bad": True})
            continue
        tid = int(row["trial_id"])
        by_trial.setdefault(tid, []).append({
            "t_rel": float(row.get("t_rel", 0.0)),
            "pred": int(row.get("pred", 0)),
            "p_max": float(row.get("p_max", 0.0)),
            "gated": bool(row.get("gated", False)),
            "p_three": row.get("p_three"),
            "win_start_rel": row.get("win_start_rel"),
            "win_end_rel": row.get("win_end_rel"),
        })
    for tid in by_trial:
        by_trial[tid].sort(key=lambda x: x.get("t_rel", 0.0))
    return by_trial


def replay_trial(trial_id: int, label: int, judgments: List[dict]) -> dict:
    tracker = MiTrialTracker(label)
    signal_bad_ticks = 0
    good_ticks = 0
    for j in judgments:
        if j.get("signal_bad"):
            signal_bad_ticks += 1
            continue
        good_ticks += 1
        tracker.add_window(float(j.get("t_rel", 0.0)), j)
    signal_bad_trial = bool(good_ticks == 0 and signal_bad_ticks > 0)
    out = tracker.finalize(signal_bad_trial=signal_bad_trial)
    out["trial_id"] = trial_id
    return out


def replay_session(events_path: Path) -> List[dict]:
    judges = load_judge_rows(events_path)
    labels: Dict[int, int] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ev = row.get("event")
        tid = row.get("trial_id")
        if tid is None:
            continue
        tid = int(tid)
        if ev in ("trial_start", "cue") and "label" in row:
            labels[tid] = int(row["label"])

    results = []
    for tid, jrows in sorted(judges.items()):
        lab = labels.get(tid)
        if lab is None:
            continue
        results.append(replay_trial(tid, lab, jrows))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="MI 多数票离线计分回放")
    ap.add_argument("events_jsonl")
    args = ap.parse_args()
    path = Path(args.events_jsonl)
    rows = replay_session(path)
    n_valid = sum(1 for r in rows if r.get("valid"))
    n_correct = sum(1 for r in rows if r.get("correct"))
    print(f"{path.name}: {len(rows)} trials, valid={n_valid}, correct={n_correct}")
    for r in rows:
        print(
            f"  trial {r['trial_id']:3d}  score={r['score']:.0f}  "
            f"valid={r['valid']}  correct={r.get('correct')}  "
            f"pred={r.get('pred')}  reason={r.get('invalid_reason') or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
