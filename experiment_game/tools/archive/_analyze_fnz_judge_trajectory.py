"""fnz 三 session：判定轨迹、primary_judge、引导效应。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "experiment_game" / "data" / "sessions"
SESSIONS = [
    "fnz_ws01_20260826_164149",
    "fnz_ws02_20260826_171537",
    "fnz_ws03_20260826_174526",
]
LABELS = {0: "Rest", 1: "Left", 2: "Right"}
PRIMARY_S = 4.0


def load_rows(sess: str):
    p = ROOT / sess / "v3_trial_features.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def analyze_session(sess: str, rows):
    print(f"\n{'='*60}\n{sess}\n{'='*60}")
    if not rows:
        print("  (no v3_trial_features)"); return
    n = len(rows)
    valid = sum(1 for r in rows if r.get("valid"))
    pj_ok = sum(1 for r in rows if (r.get("primary_judge") or {}).get("pred") == r["label"])
    print(f"trials={n} valid={valid} primary_acc={pj_ok}/{n}={pj_ok/n:.1%}")

    # pred at each t_rel majority
    by_t = defaultdict(list)
    flip_end = 0
    flip_any = 0
    for r in rows:
        js = [j for j in r.get("judgments") or [] if not j.get("signal_bad")]
        if not js:
            continue
        preds = [j["pred"] for j in js]
        lab = r["label"]
        primary = (r.get("primary_judge") or {}).get("pred")
        # was correct at any point during MI?
        any_ok = lab in preds
        last = js[-1]["pred"]
        if any_ok and primary != lab:
            flip_any += 1
        if len(js) >= 2 and js[-2]["pred"] == lab and primary != lab:
            flip_end += 1
        for j in js:
            by_t[round(float(j["t_rel"]), 1)].append((j["pred"] == lab, j["pred"]))

    print(f"  ever_correct_during_MI: {sum(1 for r in rows if lab_ok_any(r))}/{n}")
    print(f"  primary_wrong_but_sometime_right: {flip_any}")
    print(f"  right_at_penultimate_wrong_primary: {flip_end}")

    for cond in ("guided", "no_guide"):
        sub = [r for r in rows if r.get("cond") == cond]
        if not sub:
            continue
        acc = sum(1 for r in sub if (r.get("primary_judge") or {}).get("pred") == r["label"]) / len(sub)
        v = sum(1 for r in sub if r.get("valid")) / len(sub)
        print(f"  {cond}: n={len(sub)} primary_acc={acc:.1%} valid_rate={v:.1%}")

    # score vs valid
    scores = [r.get("score") for r in rows if r.get("score") is not None]
    if scores:
        print(f"  score mean={np.mean(scores):.2f} valid_scores>={4}: {sum(1 for s in scores if s>=4)}/{len(scores)}")


def lab_ok_any(r):
    lab = r["label"]
    for j in r.get("judgments") or []:
        if not j.get("signal_bad") and j.get("pred") == lab:
            return True
    return False


def main():
    for sess in SESSIONS:
        analyze_session(sess, load_rows(sess))

    # ws03 detailed flip examples
    rows = load_rows("fnz_ws03_20260826_174526")
    print(f"\n{'='*60}\nws03 trials: correct mid-MI but wrong @ primary (t=4.0s)\n{'='*60}")
    for r in rows:
        js = [j for j in r.get("judgments") or [] if not j.get("signal_bad")]
        if not js:
            continue
        lab = r["label"]
        pj = (r.get("primary_judge") or {}).get("pred")
        preds = [LABELS[j["pred"]] for j in js]
        if lab_ok_any(r) and pj != lab:
            ts = [j["t_rel"] for j in js]
            print(f"  T{r['trial_id']:02d} label={LABELS[lab]} primary={LABELS.get(pj)} preds@t={list(zip(ts,preds))}")


if __name__ == "__main__":
    main()
