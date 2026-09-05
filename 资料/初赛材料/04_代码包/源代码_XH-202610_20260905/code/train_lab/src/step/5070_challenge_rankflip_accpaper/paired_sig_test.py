# -*- coding: utf-8 -*-
"""轨 S 附证：决赛候选 vs S0 的 6 折配对显著性（Wilcoxon + 配对 t）。

p 值不自动换主交卷；只写 JSON / 供登记表引用。

用法：
  python paired_sig_test.py --replay-json path/to/replay_FM_latest.json
  python paired_sig_test.py --replay-json ... --decision-json path/to/decision_latest.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from exp35_config import exp35_out
from make_submission_candidates import build_candidates, pick_structural_arms


def _fold_accs(replay: dict, arm_id: str) -> list[float]:
    arm = (replay.get("results") or {}).get(arm_id) or {}
    folds = arm.get("folds") or []
    return [float(fr["val_metrics"]["acc"]) for fr in sorted(folds, key=lambda x: int(x["fold"]))]


def _wilcoxon_two_sided(diff: list[float]) -> float:
    """Signed-rank; ties (diff==0) dropped. Fallback if scipy missing."""
    d = [x for x in diff if x != 0.0]
    n = len(d)
    if n < 1:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        # zero_method='wilcox' drops zeros (already dropped)
        res = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto")
        return float(res.pvalue)
    except Exception:
        # Exact small-n permutation of signs (n<=12)
        abs_d = [abs(x) for x in d]
        ranks = _rankdata(abs_d)
        w_plus = sum(r for x, r in zip(d, ranks) if x > 0)
        # two-sided via all 2^n sign assignments
        from itertools import product

        null = []
        for signs in product((-1.0, 1.0), repeat=n):
            null.append(sum(s * r for s, r in zip(signs, ranks) if s > 0))
        # Actually W+ under random signs: sum of ranks where sign is +
        null_wp = []
        for signs in product((-1.0, 1.0), repeat=n):
            null_wp.append(sum(r for s, r in zip(signs, ranks) if s > 0))
        extreme = abs(w_plus - 0.5 * sum(ranks))
        p = sum(1 for v in null_wp if abs(v - 0.5 * sum(ranks)) >= extreme - 1e-12) / len(null_wp)
        return float(p)


def _rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _paired_t_pvalue(diff: list[float]) -> float:
    n = len(diff)
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import ttest_rel
        import numpy as np

        a = np.asarray(diff, dtype=float)
        # ttest_rel needs two samples; use zeros as baseline
        res = ttest_rel(a, np.zeros_like(a))
        return float(res.pvalue)
    except Exception:
        mean = sum(diff) / n
        var = sum((x - mean) ** 2 for x in diff) / (n - 1)
        if var <= 0:
            return 0.0 if mean != 0 else float("nan")
        t = mean / math.sqrt(var / n)
        # crude two-sided via normal approx for n~6
        z = abs(t)
        # Φ(-z)*2 approx
        p = math.erfc(z / math.sqrt(2.0))
        return float(p)


def _bootstrap_ci(diff: list[float], n_boot: int = 5000, seed: int = 35) -> list[float]:
    if not diff:
        return [float("nan"), float("nan")]
    import random

    rng = random.Random(seed)
    n = len(diff)
    means = []
    for _ in range(n_boot):
        sample = [diff[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return [float(lo), float(hi)]


def compare_arm(replay: dict, cand_id: str, arm_id: str, s0_arm: str = "F0") -> dict:
    a = _fold_accs(replay, arm_id)
    b = _fold_accs(replay, s0_arm)
    if len(a) != len(b) or not a:
        return {
            "candidate_id": cand_id,
            "source_arm": arm_id,
            "error": f"fold_len mismatch {len(a)} vs {len(b)}",
        }
    diff = [x - y for x, y in zip(a, b)]
    mean_diff = sum(diff) / len(diff)
    return {
        "candidate_id": cand_id,
        "source_arm": arm_id,
        "s0_arm": s0_arm,
        "fold_acc_cand": a,
        "fold_acc_s0": b,
        "fold_diff": diff,
        "mean_diff": mean_diff,
        "wilcoxon_p": _wilcoxon_two_sided(diff),
        "paired_t_p": _paired_t_pvalue(diff),
        "bootstrap_ci95_mean_diff": _bootstrap_ci(diff),
        "alpha": 0.05,
        "note": "p 不自动换主交卷",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-json", type=Path, required=True)
    ap.add_argument("--decision-json", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    replay = json.loads(args.replay_json.read_text(encoding="utf-8"))
    cands = build_candidates(replay)
    if args.decision_json and args.decision_json.is_file():
        dec = json.loads(args.decision_json.read_text(encoding="utf-8"))
        cands = dec.get("candidates") or cands

    arms = pick_structural_arms(replay.get("results") or {})
    comparisons = []
    for c in cands:
        if c["id"] == "S0":
            continue
        comparisons.append(compare_arm(replay, c["id"], c["source_arm"]))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp35_out() / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scheme": "Exp35 v0.2 §9.4",
        "replay_json": str(args.replay_json),
        "structural_arms": arms,
        "candidates": cands,
        "comparisons": comparisons,
        "decision_rule": "Wilcoxon/p 仅附证；主交卷仍按 Val argmax + §12.1",
    }
    out = out_dir / f"paired_sig_{stamp}.json"
    latest = out_dir / "paired_sig_latest.json"
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    out.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")

    print("structural_arms", arms)
    for c in comparisons:
        if "error" in c:
            print(c["candidate_id"], c["error"])
            continue
        sig = (c["wilcoxon_p"] == c["wilcoxon_p"]) and (c["wilcoxon_p"] < args.alpha)
        print(
            f"{c['candidate_id']}({c['source_arm']}) mean_diff={c['mean_diff']:+.4f} "
            f"wilcoxon_p={c['wilcoxon_p']:.4f} paired_t_p={c['paired_t_p']:.4f} "
            f"ci95={c['bootstrap_ci95_mean_diff']} sig@0.05={sig}"
        )
    print("wrote", out)
    print("wrote", latest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
