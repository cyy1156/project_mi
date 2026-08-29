#!/usr/bin/env python3
"""OpenBMI dump：经典双向平滑 vs 流式 S vs 因果 C（冻结 E1f 融合）。

零训练。融合超参取 replay_e1f.json（方案 26）。

用法::

    python replay_classic_vs_causal.py
    python replay_classic_vs_causal.py --tau 0.4 --tau-grid 0.4,0.45,0.5,0.55,0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "code"))

from e1_fusion_core import (  # noqa: E402
    _trial_order,
    acc_paper_for_split,
    fuse_pipeline,
    simulate_conf_early_stop,
)
from s26_config import DEFAULT_MEMBERS  # noqa: E402
from prob_io import load_members  # noqa: E402

from adapt_engine.readout import (  # noqa: E402
    majority_vote_from_probs,
    streaming_conf_stop_C,
    streaming_conf_stop_S,
)

E1F_JSON = HERE / "replay_e1f.json"
OUT_DEFAULT = HERE / "replay_classic_vs_causal.json"


def _load_e1f_cfg() -> Dict[str, Any]:
    blob = json.loads(E1F_JSON.read_text(encoding="utf-8"))
    return blob["config"]


def _t_rels_from_idxs(data: dict, idxs: List[int]) -> List[float]:
    """窗尾相对 Cue 的代理时刻：t0_sec + 3.0（3s 窗）。"""
    return [float(data["t0_sec"][i]) + 3.0 for i in idxs]


def _broadcast_trial_preds(
    base: dict,
    *,
    arm: str,
    tau: float,
) -> Tuple[dict, List[float], List[bool]]:
    """在未平滑融合概率上按试次跑臂，广播 pred 到窗，返回 t_dec 列表。"""
    out = dict(base)
    pred = np.array(base["pred"], dtype=np.int64, copy=True)
    t_decs: List[float] = []
    earlys: List[bool] = []
    buckets = _trial_order(base)
    for idxs in buckets.values():
        probs = [base["probs"][i] for i in idxs]
        t_rels = _t_rels_from_idxs(base, idxs)
        if arm == "S":
            d = streaming_conf_stop_S(probs, t_rels=t_rels, tau_conf=tau)
        elif arm == "C":
            d = streaming_conf_stop_C(probs, t_rels=t_rels, tau_conf=tau, min_windows=3)
        elif arm == "W":
            d = majority_vote_from_probs(probs, t_rels=t_rels)
        else:
            raise ValueError(arm)
        for i in idxs:
            pred[i] = int(d["pred"])
        t_decs.append(float(d["t_dec"]))
        earlys.append(bool(d.get("early")))
    out["pred"] = pred
    out["p_max"] = base["probs"].max(axis=1).astype(np.float32)
    return out, t_decs, earlys


def classic_offline_on_fused(base_raw: dict, *, tau: float) -> Tuple[dict, List[float]]:
    """方案 26：先双向 r=1 平滑，再 conf_stop。t_dec≈中心+1 到达（在线等价）。"""
    from e1_fusion_core import pack_fused, temporal_smooth

    probs_s = temporal_smooth(base_raw["probs"], base_raw, 1)
    packed = pack_fused(base_raw, probs_s)
    stopped = simulate_conf_early_stop(packed, tau_conf=tau)
    t_decs: List[float] = []
    buckets = _trial_order(base_raw)
    for idxs in buckets.values():
        picked_j = len(idxs) - 1
        for j, i in enumerate(idxs):
            if float(probs_s[i].max()) >= float(tau):
                picked_j = j
                break
        arrive_j = min(picked_j + 1, len(idxs) - 1)
        t_decs.append(float(base_raw["t0_sec"][idxs[arrive_j]]) + 3.0)
    return stopped, t_decs


def eval_arm(
    fused_raw: dict,
    *,
    arm: str,
    tau: float,
) -> Dict[str, Any]:
    if arm == "classic":
        data, t_decs = classic_offline_on_fused(fused_raw, tau=tau)
        early_frac = float(np.mean([t < 3.99 for t in t_decs])) if t_decs else None
    else:
        data, t_decs, earlys = _broadcast_trial_preds(fused_raw, arm=arm, tau=tau)
        early_frac = float(np.mean(earlys)) if earlys else None
    return {
        "val_acc_paper": acc_paper_for_split(data, "val"),
        "test_acc_paper": acc_paper_for_split(data, "test"),
        "t_dec_mean_all": float(np.mean(t_decs)) if t_decs else None,
        "early_frac_all": early_frac,
        "n_trials_timed": len(t_decs),
    }


def select_tau(
    fused_raw: dict,
    *,
    tau_grid: Sequence[float],
    mode: str = "mean_SC",
) -> Tuple[float, Dict[str, float]]:
    best_tau = float(tau_grid[0])
    best = -1.0
    scores: Dict[str, float] = {}
    for tau in tau_grid:
        if mode == "classic":
            score = eval_arm(fused_raw, arm="classic", tau=float(tau))["val_acc_paper"]
        else:
            s = eval_arm(fused_raw, arm="S", tau=float(tau))["val_acc_paper"]
            c = eval_arm(fused_raw, arm="C", tau=float(tau))["val_acc_paper"]
            score = 0.5 * (s + c)
        scores[f"{float(tau):.2f}"] = float(score)
        if score > best:
            best = score
            best_tau = float(tau)
    return best_tau, scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--tau-grid", default="0.4,0.45,0.5,0.55,0.6")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    tau_grid = [float(x) for x in args.tau_grid.split(",") if x.strip()]

    cfg = _load_e1f_cfg()
    temps = list(cfg["temperatures"])
    weights = tuple(float(w) for w in cfg["weights"])
    print("[load] four-member dumps …")
    runs = [
        DEFAULT_MEMBERS.shallow,
        DEFAULT_MEMBERS.t_shallow,
        DEFAULT_MEMBERS.eegnet,
        DEFAULT_MEMBERS.conformer,
    ]
    members = load_members(runs)
    print(f"[fuse] n_windows={len(members[0]['y'])} temps/weights frozen from E1f")
    fused_raw = fuse_pipeline(
        members,
        temperatures=temps,
        weights=weights,
        smooth_radius=0,  # 平滑交给各臂
    )

    arms = ("W", "classic", "S", "C")
    at_fixed: Dict[str, Any] = {}
    print(f"\n=== tau={args.tau} ===")
    for arm in arms:
        ev = eval_arm(fused_raw, arm=arm, tau=args.tau)
        at_fixed[arm] = ev
        print(
            f"  {arm:8s} val={ev['val_acc_paper']:.4f} test={ev['test_acc_paper']:.4f} "
            f"t_dec={ev['t_dec_mean_all']:.3f} early={ev['early_frac_all']:.1%}"
        )

    print("\n[select] shared tau* on val (mean Acc_S, Acc_C) …")
    tau_star, val_scores = select_tau(fused_raw, tau_grid=tau_grid, mode="mean_SC")
    print(f"  tau*={tau_star} scores={val_scores}")
    at_star: Dict[str, Any] = {}
    print(f"\n=== tau*={tau_star} ===")
    for arm in arms:
        ev = eval_arm(fused_raw, arm=arm, tau=tau_star)
        at_star[arm] = ev
        print(
            f"  {arm:8s} val={ev['val_acc_paper']:.4f} test={ev['test_acc_paper']:.4f} "
            f"t_dec={ev['t_dec_mean_all']:.3f} early={ev['early_frac_all']:.1%}"
        )

    # sanity: classic vs S test delta
    d_cs = at_fixed["classic"]["test_acc_paper"] - at_fixed["S"]["test_acc_paper"]
    d_cc = at_fixed["classic"]["test_acc_paper"] - at_fixed["C"]["test_acc_paper"]
    d_sc = at_fixed["S"]["test_acc_paper"] - at_fixed["C"]["test_acc_paper"]

    out = {
        "protocol": "openbmi_3s_hop100_dump · E1f fuse frozen · classic vs S vs C",
        "e1f_config": cfg,
        "member_runs": [str(r) for r in runs],
        "anchor_e1f_test": 0.6173456790123457,
        "tau_fixed": args.tau,
        "tau_star": tau_star,
        "val_tau_scores_mean_SC": val_scores,
        "at_tau_fixed": at_fixed,
        "at_tau_star": at_star,
        "deltas_test_tau_fixed_pp": {
            "classic_minus_S": d_cs * 100,
            "classic_minus_C": d_cc * 100,
            "S_minus_C": d_sc * 100,
        },
        "deltas_test_tau_star_pp": {
            "classic_minus_S": (
                at_star["classic"]["test_acc_paper"] - at_star["S"]["test_acc_paper"]
            )
            * 100,
            "classic_minus_C": (
                at_star["classic"]["test_acc_paper"] - at_star["C"]["test_acc_paper"]
            )
            * 100,
            "S_minus_C": (
                at_star["S"]["test_acc_paper"] - at_star["C"]["test_acc_paper"]
            )
            * 100,
        },
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
