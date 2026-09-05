#!/usr/bin/env python3
"""按 F5（因果平滑 + 多数票）回放 syj0828 / fnz0828 的 v3 会话。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from experiment_game.experiment.trial_scoring import (  # noqa: E402
    PRE_CUE_REST_POINTS,
    MiTrialTracker,
)

SUBJECTS = ["syj0828", "fnz0828"]
ROOT = _REPO / "experiment_game" / "data" / "subjects"
OUT = ROOT / "_f5_reeval_20260829.json"


def _phase_mode(session: Path) -> str:
    rc = session / "run_config.json"
    if not rc.is_file():
        return "?"
    blob = json.loads(rc.read_text(encoding="utf-8"))

    def dig(o):
        if isinstance(o, dict):
            if "phase_mode" in o:
                return str(o["phase_mode"])
            for v in o.values():
                r = dig(v)
                if r:
                    return r
        return None

    return dig(blob) or "?"


def _load_mi_windows(session: Path) -> dict[int, tuple[int, list]]:
    """trial_id -> (label, judgments sorted). Prefer v3_trial_features."""
    feat = session / "v3_trial_features.jsonl"
    out: dict[int, tuple[int, list]] = {}
    if feat.is_file():
        for line in feat.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = int(r["trial_id"])
            lab = int(r["label"])
            js = [j for j in (r.get("judgments") or []) if not j.get("signal_bad")]
            js = sorted(js, key=lambda j: float(j.get("t_rel", 0.0)))
            out[tid] = (lab, js)
        return out
    # fallback events
    ev = session / "events.jsonl"
    if not ev.is_file():
        return out
    labels: dict[int, int] = {}
    by: dict[int, list] = {}
    for line in ev.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") == "trial_start" and r.get("trial_id") is not None:
            labels[int(r["trial_id"])] = int(r.get("label", -1))
        if r.get("event") != "judge":
            continue
        phase = str(r.get("score_phase") or r.get("role") or "mi")
        if phase == "pre_cue_rest":
            continue
        tid = int(r["trial_id"])
        by.setdefault(tid, []).append(r)
    for tid, js in by.items():
        js = sorted(js, key=lambda j: float(j.get("t_rel", 0.0)))
        out[tid] = (labels.get(tid, -1), js)
    return out


def _load_rest_windows(session: Path) -> dict[int, list]:
    ev = session / "events.jsonl"
    by: dict[int, list] = {}
    if not ev.is_file():
        return by
    for line in ev.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") != "judge":
            continue
        phase = str(r.get("score_phase") or r.get("role") or "")
        if phase != "pre_cue_rest":
            continue
        if r.get("signal_bad"):
            continue
        tid = int(r["trial_id"])
        by.setdefault(tid, []).append(r)
    for tid in by:
        by[tid] = sorted(by[tid], key=lambda j: float(j.get("t_rel", 0.0)))
    return by


def eval_session(session: Path) -> dict:
    mi = _load_mi_windows(session)
    rest = _load_rest_windows(session)
    mi_ok = mi_n = 0
    rest_ok = rest_n = 0
    score = 0.0
    pred_hist = Counter()
    conf = Counter()  # (y, pred)
    old_score_ok = 0
    old_score_n = 0

    # old score from features if present
    feat = session / "v3_trial_features.jsonl"
    old_by_tid = {}
    if feat.is_file():
        for line in feat.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            old_by_tid[int(r["trial_id"])] = r

    for tid, (lab, js) in sorted(mi.items()):
        if lab not in (1, 2) or not js:
            continue
        tr = MiTrialTracker(lab)
        for j in js:
            tr.add_window(float(j.get("t_rel", 0.0)), j)
        s = tr.finalize()
        mi_n += 1
        pred = int(s["pred"]) if s.get("pred") is not None else -1
        pred_hist[pred] += 1
        conf[(lab, pred)] += 1
        if s["correct"]:
            mi_ok += 1
            score += float(s["score"])
        old = old_by_tid.get(tid)
        if old is not None and old.get("score") is not None:
            old_score_n += 1
            if bool(old.get("score", 0) > 0) == bool(s["correct"]):
                old_score_ok += 1

    for tid, js in rest.items():
        if not js:
            continue
        tr = MiTrialTracker(0, correct_points=PRE_CUE_REST_POINTS)
        for j in js:
            tr.add_window(float(j.get("t_rel", 0.0)), j)
        s = tr.finalize()
        rest_n += 1
        if s["correct"]:
            rest_ok += 1
            score += float(s["score"])

    return {
        "session": session.name,
        "mi_n": mi_n,
        "mi_acc": (mi_ok / mi_n) if mi_n else None,
        "mi_correct": mi_ok,
        "rest_n": rest_n,
        "rest_acc": (rest_ok / rest_n) if rest_n else None,
        "rest_correct": rest_ok,
        "session_points_f5": score,
        "pred_hist": {str(k): int(v) for k, v in pred_hist.items()},
        "confusion_lr": {
            f"y{y}->p{p}": int(c) for (y, p), c in sorted(conf.items())
        },
        "agree_old_score_frac": (old_score_ok / old_score_n) if old_score_n else None,
        "n_compared_old_score": old_score_n,
    }


def main() -> int:
    report = {
        "rule": "F5 causal_smooth_majority",
        "channel_order_expected": ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"],
        "subjects": {},
    }
    print("F5 re-eval (causal smooth + majority)")
    for sid in SUBJECTS:
        sroot = ROOT / sid / "sessions"
        rows = []
        print("=" * 64, sid)
        for sp in sorted(sroot.iterdir()):
            if not sp.is_dir() or sp.name.startswith("_"):
                continue
            if _phase_mode(sp) != "v3_session":
                print(f"  SKIP {sp.name} (not v3)")
                continue
            ev = eval_session(sp)
            rows.append(ev)
            print(
                f"  {sp.name}: MI {ev['mi_correct']}/{ev['mi_n']}="
                f"{ev['mi_acc']:.1%}  Rest {ev['rest_correct']}/{ev['rest_n']} "
                f"pts={ev['session_points_f5']:.1f} "
                f"agree_old={ev['agree_old_score_frac']}"
            )
            print(f"    pred_hist={ev['pred_hist']} conf={ev['confusion_lr']}")
        if rows:
            mi_c = sum(r["mi_correct"] for r in rows)
            mi_n = sum(r["mi_n"] for r in rows)
            report["subjects"][sid] = {
                "n_sessions": len(rows),
                "mi_acc_pooled": mi_c / mi_n if mi_n else None,
                "mi_correct": mi_c,
                "mi_n": mi_n,
                "sessions": rows,
            }
            print(
                f"  POOLED MI: {mi_c}/{mi_n}={mi_c/mi_n:.1%}" if mi_n else "  POOLED empty"
            )
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
