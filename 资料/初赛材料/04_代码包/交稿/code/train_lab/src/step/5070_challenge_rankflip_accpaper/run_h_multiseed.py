# -*- coding: utf-8 -*-
"""轨 H 多种子裁决 Q4（Exp35 v0.2.1 · P1）。

流程：
  1) H0 × seeds 估 run-to-run σ（相对 Exp34 shallow 锚）
  2) H1–H5 × 同种子满 6 折
  3) 按 §10.3：最佳 Hx 与 conformer 锚差 <0.020 → H-a；≥0.030 → H-b

用法：
  python run_h_multiseed.py
  python run_h_multiseed.py --seeds 42,43,44 --arms H0
  python run_h_multiseed.py --seeds 42,43,44 --arms H0,H1,H2,H3,H4,H5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

from exp35_config import ANCHOR_CONFORMER_A59, ANCHOR_SHALLOW_A59, exp35_out, scheme_doc
from run_shallow_recipe_h import run_arm


def _adjudicate(by_arm: dict[str, list[dict]]) -> dict:
    # 每臂：跨种子 mean of mean
    arm_stats = {}
    for arm, runs in by_arm.items():
        means = [float(r["val_acc_mean"]) for r in runs if r.get("val_acc_mean") is not None]
        if not means:
            continue
        m = float(np.mean(means))
        s = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        arm_stats[arm] = {
            "n_seeds": len(means),
            "val_acc_mean_of_means": m,
            "val_acc_std_across_seeds": s,
            "seed_means": means,
            "gap_vs_conformer": ANCHOR_CONFORMER_A59 - m,
            "delta_vs_shallow_anchor": m - ANCHOR_SHALLOW_A59,
        }

    h0 = arm_stats.get("H0", {})
    others = {k: v for k, v in arm_stats.items() if k != "H0"}
    best_arm, best = None, None
    for arm, st in others.items():
        if best is None or st["val_acc_mean_of_means"] > best["val_acc_mean_of_means"]:
            best_arm, best = arm, st

    verdict = "inconclusive"
    note = ""
    if best is not None:
        gap = best["gap_vs_conformer"]
        if gap < 0.020:
            verdict = "H-a_underfit"
            note = f"{best_arm} 距 conformer 锚 {gap:.3f} < 0.020"
        elif gap >= 0.030:
            verdict = "H-b_domain_mismatch"
            note = f"{best_arm} 距 conformer 锚 {gap:.3f} ≥ 0.030"
        else:
            verdict = "gray_zone"
            note = f"{best_arm} 距 conformer 锚 {gap:.3f} ∈ [0.020, 0.030)"

    h0_drift = None
    if h0:
        h0_drift = {
            "mean_vs_anchor": h0["delta_vs_shallow_anchor"],
            "seed_sigma": h0["val_acc_std_across_seeds"],
            "repro_band_ok": abs(h0["delta_vs_shallow_anchor"]) <= 0.01
            or (h0["val_acc_std_across_seeds"] >= abs(h0["delta_vs_shallow_anchor"]) - 0.005),
            "note": "若 seed σ 能解释 +2.7pp 漂移则复现链可接受；否则查训练链差异",
        }

    return {
        "arm_stats": arm_stats,
        "best_non_h0": {"arm": best_arm, **(best or {})} if best else None,
        "verdict": verdict,
        "note": note,
        "h0_repro": h0_drift,
        "anchors": {"shallow": ANCHOR_SHALLOW_A59, "conformer": ANCHOR_CONFORMER_A59},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--arms", default="H0,H1,H2,H3,H4,H5")
    ap.add_argument("--max-folds", type=int, default=0, help="0=满 6 折")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    tag = args.run_tag.strip() or f"multiseed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    max_folds = args.max_folds  # 0 → run_arm 里 fold>=0 永不 break? Check: if max_folds > 0 and fold >= max_folds. So 0 means all folds. Good.

    by_arm: dict[str, list[dict]] = {a: [] for a in arms}
    out_root = exp35_out() / "H" / tag
    out_root.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        for seed in seeds:
            print(f"=== {arm} seed={seed} ===", flush=True)
            summary = run_arm(arm, max_folds, f"{tag}_{arm}_s{seed}", seed=seed)
            by_arm[arm].append(summary)
            # 渐进写盘
            mid = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_tag": tag,
                "completed": {a: len(by_arm[a]) for a in arms},
                "adjudication": _adjudicate(by_arm),
            }
            (out_root / "progress.json").write_text(
                json.dumps(mid, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    adj = _adjudicate(by_arm)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": tag,
        "seeds": seeds,
        "arms": arms,
        "runs": {a: by_arm[a] for a in arms},
        "adjudication": adj,
    }
    out = out_root / "multiseed_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = exp35_out() / "H" / "multiseed_latest.json"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")

    # 简表写进方案总结
    lines = [
        f"# 轨 H 多种子裁决（{tag}）",
        "",
        f"> 生成：{report['generated_at']} · seeds={seeds}",
        "",
        f"**裁决：`{adj['verdict']}`** — {adj.get('note','')}",
        "",
        "| 臂 | n_seeds | mean±σ(across seeds) | vs shallow 锚 | vs conformer 锚 |",
        "|----|---------|----------------------|---------------|-----------------|",
    ]
    for arm, st in (adj.get("arm_stats") or {}).items():
        lines.append(
            f"| {arm} | {st['n_seeds']} | {st['val_acc_mean_of_means']:.3f}±{st['val_acc_std_across_seeds']:.3f} | "
            f"{st['delta_vs_shallow_anchor']:+.3f} | {st['gap_vs_conformer']:+.3f} |"
        )
    if adj.get("h0_repro"):
        hr = adj["h0_repro"]
        lines += [
            "",
            "## H0 复现",
            "",
            f"- mean vs Exp34 shallow 锚：{hr['mean_vs_anchor']:+.3f}",
            f"- across-seed σ：{hr['seed_sigma']:.3f}",
            f"- repro_band_ok（启发式）：{hr['repro_band_ok']}",
        ]
    doc = scheme_doc() / "总结" / f"轨H_多种子裁决_{tag}.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (scheme_doc() / "总结" / "轨H_多种子裁决_最新.md").write_text(doc.read_text(encoding="utf-8"), encoding="utf-8")

    print("adjudication", adj.get("verdict"), adj.get("note"))
    print("wrote", out)
    print("wrote", latest)
    print("wrote", doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
