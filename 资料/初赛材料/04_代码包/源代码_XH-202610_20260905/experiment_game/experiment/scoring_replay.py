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


def load_judge_rows(
    events_path: Path,
    *,
    score_phase: Optional[str] = "mi",
) -> Dict[int, List[dict]]:
    """trial_id → 按 t_rel 排序的 judge 行。

    score_phase:
      - \"mi\"（默认）：只取 MI 判定（排除 Cue 前静息）
      - \"pre_cue_rest\"：只取 Cue 前静息判定
      - None：全部判定（会混 MI+Rest，一般勿用于多数票回放）
    """
    by_trial: Dict[int, List[dict]] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "judge":
            continue
        phase = str(row.get("score_phase") or row.get("role") or "mi")
        if phase == "pre_cue_rest":
            phase = "pre_cue_rest"
        else:
            phase = "mi"
        if score_phase is not None and phase != score_phase:
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
            "score_phase": phase,
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
    from experiment_game.experiment.trial_scoring import PRE_CUE_REST_POINTS

    judges = load_judge_rows(events_path, score_phase="mi")
    rest_judges = load_judge_rows(events_path, score_phase="pre_cue_rest")
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
        mi = replay_trial(tid, lab, jrows)
        rest_rows = rest_judges.get(tid) or []
        if rest_rows:
            rest = replay_trial(tid, 0, rest_rows)
            rest["score"] = (
                float(PRE_CUE_REST_POINTS) if rest.get("correct") else 0.0
            )
            rest["correct_points"] = PRE_CUE_REST_POINTS
            mi["rest_score"] = rest["score"]
            mi["rest_correct"] = rest.get("correct")
            mi["session_points"] = float(mi.get("score") or 0.0) + float(rest["score"])
        else:
            mi["rest_score"] = None
            mi["session_points"] = float(mi.get("score") or 0.0)
        results.append(mi)
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
