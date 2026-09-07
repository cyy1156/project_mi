"""方案 23 · 被试级配对统计（Δ、CI、Wilcoxon）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_per_subject_acc(run_dir: Path) -> dict[str, float]:
    """从 summary/fold metrics 聚合 per-subject test acc（若无则按 fold 近似）。"""
    summ = run_dir / "summary.json"
    if not summ.is_file():
        raise FileNotFoundError(summ)
    s = json.loads(summ.read_text(encoding="utf-8"))
    ps = s.get("per_subject_test_acc_paper")
    if ps:
        return {str(k): float(v) for k, v in ps.items()}
    # fallback: fold-level only
    folds = s.get("folds", [])
    out: dict[str, float] = {}
    for i, f in enumerate(folds):
        out[f"fold{i}"] = float(f.get("test_acc_paper", f.get("test_trial_metrics", {}).get("acc_paper", 0)))
    return out


def paired_stats(
    run_a: Path,
    run_b: Path,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """run_b − run_a 的 per-key 配对差（key 对齐）。"""
    sa = _load_per_subject_acc(run_a)
    sb = _load_per_subject_acc(run_b)
    keys = sorted(set(sa) & set(sb))
    if not keys:
        return {"n_pairs": 0, "error": "no overlapping keys"}
    diffs = np.array([sb[k] - sa[k] for k in keys], dtype=np.float64)
    mean = float(diffs.mean())
    std = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    n = len(diffs)
    ci_half = 1.96 * std / max(n**0.5, 1.0)
    win_rate = float((diffs > 0).mean())
    p_val = float("nan")
    try:
        from scipy.stats import wilcoxon

        if n >= 2 and np.any(diffs != 0):
            p_val = float(wilcoxon(diffs).pvalue)
    except Exception:
        pass
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": n,
        "mean_delta_b_minus_a": mean,
        "std_delta": std,
        "ci95_low": mean - ci_half,
        "ci95_high": mean + ci_half,
        "win_rate_b_gt_a": win_rate,
        "wilcoxon_p": p_val,
        "keys": keys,
        "diffs_pp": (diffs * 100).tolist(),
    }


def compare_runs_cli(run_a: str, run_b: str, *, name_a: str = "", name_b: str = "") -> None:
    stats = paired_stats(
        Path(run_a),
        Path(run_b),
        label_a=name_a or Path(run_a).name,
        label_b=name_b or Path(run_b).name,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Scheme-23 paired subject stats")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--name-a", default="")
    p.add_argument("--name-b", default="")
    args = p.parse_args()
    compare_runs_cli(args.run_a, args.run_b, name_a=args.name_a, name_b=args.name_b)
