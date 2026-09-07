# -*- coding: utf-8 -*-
"""Exp40：组合 MC∘TTA（若单臂进候选）+ §5 终态裁决。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from exp40_config import (
    FOLD_OK_MIN,
    RB8_ANCHOR,
    TIE_PP,
    exp39_out,
    exp40_out,
)
from replay_margin_b8 import _apply_bias, _fit_bias, _split_folds, fold_ok


COMPLEXITY = {
    "R-B8_raw": 1,
    "TTA-B8": 2,
    "MC-B8": 3,
    "MC∘TTA": 4,
    "MC+TTA": 4,
}


def _accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y.astype(int)).mean())


def run_mc_on_tta(tta_prob: np.ndarray, tta_y: np.ndarray, base_accs: list[float]) -> dict:
    probs_l, y_l = _split_folds(tta_prob, tta_y)
    n_folds = len(probs_l)
    nested_accs, nested_folds = [], []
    for i in range(n_folds):
        p_tr = np.concatenate([probs_l[j] for j in range(n_folds) if j != i], axis=0)
        y_tr = np.concatenate([y_l[j] for j in range(n_folds) if j != i], axis=0)
        b = _fit_bias(p_tr, y_tr)
        p_te = _apply_bias(probs_l[i], b)
        acc = _accuracy(p_te, y_l[i])
        nested_accs.append(acc)
        nested_folds.append({"fold": i, "bias": b.tolist(), "acc": acc})
        print(f"  MC+TTA fold{i} acc={acc:.4f}", flush=True)
    fo = fold_ok(nested_accs, base_accs)
    mean = float(np.mean(nested_accs))
    delta = mean - float(np.mean(base_accs))
    return {
        "arm_id": "MC∘TTA",
        "val_acc_mean": mean,
        "val_acc_std": float(np.std(nested_accs, ddof=1)) if n_folds > 1 else 0.0,
        "fold_accs": nested_accs,
        "folds": nested_folds,
        "delta_vs_rb8": delta,
        **fo,
    }


def distance_to_uniform(counts: dict[str, int]) -> float:
    total = sum(counts.values()) or 1
    ps = [counts[k] / total for k in ("L", "R", "Rest")]
    return float(sum(abs(p - 1.0 / 3.0) for p in ps))


def decide(arms: dict, base_mean: float) -> dict:
    """§5.1"""
    cands = ["R-B8_raw"]
    mc = arms.get("MC-B8") or {}
    tta = arms.get("TTA-B8") or {}
    combo = arms.get("MC∘TTA")

    if mc.get("enter_candidate"):
        cands.append("MC-B8")
    if tta.get("enter_candidate"):
        cands.append("TTA-B8")

    if combo is not None:
        singles_in = [a for a in ("MC-B8", "TTA-B8") if a in cands]
        if singles_in:
            floor = max(base_mean, max(float(arms[a]["val_acc_mean"]) for a in singles_in))
            if (
                float(combo["val_acc_mean"]) + 1e-12 >= floor
                and combo.get("fold_ok")
            ):
                combo["enter_candidate"] = True
                cands.append("MC∘TTA")
            else:
                combo["enter_candidate"] = False

    def mean_of(aid: str) -> float:
        if aid == "R-B8_raw":
            return base_mean
        return float(arms[aid]["val_acc_mean"])

    scored = [(aid, mean_of(aid), COMPLEXITY[aid]) for aid in cands]
    scored.sort(key=lambda x: (-x[1], x[2], x[0]))
    winner = scored[0][0]
    w_mean = scored[0][1]
    tied = [s for s in scored if (w_mean - s[1]) < TIE_PP - 1e-12]
    tied.sort(key=lambda x: (x[2], x[0]))
    final = tied[0][0]

    return {
        "candidates": cands,
        "scored": [{"id": a, "nested_mean": m, "complexity": c} for a, m, c in scored],
        "tie_pool": [t[0] for t in tied],
        "decision_engineering_final": final,
        "write_new_csv": final != "R-B8_raw",
        "label": "工程选择，非显著性结论",
    }


def main() -> int:
    replay = exp40_out() / "replay"
    ranking = json.loads((exp39_out() / "replay" / "ranking_latest.json").read_text(encoding="utf-8"))
    base_accs = ranking["arms"]["R-B8"]["fold_accs"]
    base_mean = float(ranking["arms"]["R-B8"]["val_acc_mean"])

    margin = json.loads((replay / "margin_latest.json").read_text(encoding="utf-8"))
    tta = json.loads((replay / "tta_latest.json").read_text(encoding="utf-8"))

    arms = {
        "R-B8_raw": {
            "arm_id": "R-B8_raw",
            "val_acc_mean": base_mean,
            "val_acc_std": float(ranking["arms"]["R-B8"]["val_acc_std"]),
            "fold_accs": base_accs,
            "enter_candidate": True,
        },
        "MC-B8": margin,
        "TTA-B8": tta,
    }

    need_combo = bool(margin.get("enter_candidate") or tta.get("enter_candidate"))
    if need_combo:
        print("=== T2 MC+TTA ===", flush=True)
        tta_prob = np.load(replay / "tta_b8_oof_prob.npy")
        tta_y = np.load(replay / "tta_b8_oof_y.npy")
        combo = run_mc_on_tta(tta_prob, tta_y, base_accs)
        arms["MC∘TTA"] = combo
        (replay / "combo_latest.json").write_text(
            json.dumps(combo, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print("跳过 T2（H1/T1 均未进候选）", flush=True)

    eng = decide(arms, base_mean)

    # H0 风险（方案已录入）
    h0 = {
        "hamming_s0_rb8": "54/120=45%",
        "s0_margin": {"L": 29, "R": 44, "Rest": 47},
        "rb8_margin": {"L": 38, "R": 21, "Rest": 61},
        "acc_ceiling_if_404040": {"S0": 0.908, "R-B8": 0.825},
    }
    h0["s0_dist_uniform"] = distance_to_uniform(h0["s0_margin"])
    h0["rb8_dist_uniform"] = distance_to_uniform(h0["rb8_margin"])

    # Q2：若采用 MC，用 pool bias 作用到 R-B8 test 需另算；此处仅登记「诊断锁」
    margin_note = (
        "Q2 distance_to_uniform 改善是 MC 定义效应，禁止放松 Acc/fold_ok 门槛"
    )

    if not margin.get("enter_candidate"):
        risk_mc = "Rest/类边际偏斜不可校准、属模型本性（MC 未进候选）"
    else:
        risk_mc = "MC 进候选；仍须看终态是否选中"

    doc = {
        "experiment": 40,
        "scheme_version": "v0.2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rb8_anchor": RB8_ANCHOR,
        "arms": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in arms.items()
        },
        "decision_engineering": eng,
        "h0_risk": h0,
        "risk_notes": {
            "mc": risk_mc,
            "q2": margin_note,
            "prior": "两臂大概率不采用；默认终态 R-B8_raw",
        },
        "decision_science": "KEEP_S0_discipline",
        "algorithm_freeze": True,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (replay / f"harden_{stamp}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (replay / "harden_latest.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ENGINEERING", eng["decision_engineering_final"], flush=True)
    print("wrote", replay / "harden_latest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
