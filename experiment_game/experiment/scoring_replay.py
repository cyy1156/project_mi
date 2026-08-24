"""从 events.jsonl 离线回放 D8 计分（与在线 trial_v2 同规则）。

用法：
  python -m experiment_game.experiment.scoring_replay path/to/events.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "code"))

from adapt_engine.scoring_v21 import ScoringConfig, score_trial_from_judgments  # noqa: E402
from experiment_game.experiment.v2_config import V2Config  # noqa: E402


def load_judge_rows(events_path: Path) -> Dict[int, List[dict]]:
    """trial_id → 按 t_rel 排序的 judge 行。"""
    by_trial: Dict[int, List[dict]] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "judge":
            continue
        tid = int(row["trial_id"])
        by_trial.setdefault(tid, []).append({
            "t": float(row.get("t_rel", 0.0)),
            "pred": int(row.get("pred", 0)),
            "p_max": row.get("p_max"),
            "gated": row.get("gated"),
        })
    for tid in by_trial:
        by_trial[tid].sort(key=lambda x: x["t"])
    return by_trial


def replay_trial(
    trial_id: int,
    label: int,
    judgments: List[dict],
    cfg: ScoringConfig,
    *,
    ended_early: bool = False,
    end_reason: Optional[str] = None,
) -> dict:
    verdict = score_trial_from_judgments(
        label, judgments, cfg, ended_early=ended_early, end_reason=end_reason
    )
    out = verdict.to_dict()
    out["trial_id"] = trial_id
    return out


def replay_session(events_path: Path, config_path: Optional[str] = None) -> List[dict]:
    v2cfg = V2Config.load_yaml(config_path)
    sc = v2cfg.scoring_config()
    judges = load_judge_rows(events_path)
    labels: Dict[int, int] = {}
    mi_ends: Dict[int, dict] = {}
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
        if ev == "mi_end":
            mi_ends[tid] = {
                "early": bool(row.get("early", False)),
                "reason": row.get("reason"),
            }

    results = []
    for tid, jrows in sorted(judges.items()):
        lab = labels.get(tid)
        if lab is None:
            continue
        end = mi_ends.get(tid, {})
        results.append(replay_trial(
            tid, lab, jrows, sc,
            ended_early=bool(end.get("early")),
            end_reason=end.get("reason"),
        ))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="D8 离线计分回放")
    ap.add_argument("events_jsonl")
    ap.add_argument("--config", default=None, help="v2_session.yaml 路径")
    args = ap.parse_args()
    path = Path(args.events_jsonl)
    rows = replay_session(path, args.config)
    n_valid = sum(1 for r in rows if r.get("valid"))
    print(f"{path.name}: {len(rows)} trials, valid={n_valid}")
    for r in rows:
        print(
            f"  trial {r['trial_id']:3d}  score={r['score']:.1f}  "
            f"valid={r['valid']}  early={r['early_stop']}  "
            f"reason={r.get('invalid_reason') or r.get('early_stop_reason') or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
