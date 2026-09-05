# -*- coding: utf-8 -*-
"""Exp39：收尾回放 — R-B8 / R-pool-* / R-uni50 + 双轨决策。

用法：
  python replay_closing.py
  python replay_closing.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent
from exp39_config import (  # noqa: E402
    MEMBER_KEYS,
    N_BOOT,
    PREFER_TAG,
    TIE_PP,
    UNI50_POOLED_LINE,
    a59_step,
    b8_step,
    exp37_out,
    exp37_step,
    exp38_out,
    exp39_out,
    rankflip_step,
)

_E37 = exp37_step()
_A59 = a59_step()
_B8 = b8_step()
_RF = rankflip_step()
for p in (_E37, _A59, _B8, _RF, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e1f_constrained import fit_e1f_constrained, fuse_with_config  # noqa: E402
from e1f_core import E1fConfig, accuracy  # noqa: E402
from exp35_config import FusionConstraints  # noqa: E402
from member_paths import find_all_a59_members, find_all_b8_members, n_folds_available  # noqa: E402
from metrics_secondary import (  # noqa: E402
    cluster_bootstrap_delta,
    mcnemar_exact,
    paired_t_p,
    wilcoxon_p,
)
from replay_nested import (  # noqa: E402
    _apply_e1f,
    _concat_member_probs,
    _fit_e1f,
    _pack_arm,
    load_track_bank,
    nested_e1f_track,
    save_oof,
)

# 工程复杂度：越小越简（§5.2 平手裁决）
COMPLEXITY = {
    "V1": 1,
    "F1": 1,
    "S0_csv": 2,  # 已有文件，同复杂度时优先
    "N0": 2,
    "R-B8": 3,
    "R-pool-S0": 4,
    "R-pool-B8": 4,
    "R-uni50": 5,
}


def nested_single_member(bank: dict, name: str, arm_id: str, desc: str) -> dict:
    """单成员：无自由融合参数；fold Acc = 该成员 Val Acc（温度可选：嵌套 fit T）。"""
    n_folds = bank["n_folds"]
    # 找到成员下标
    names0 = bank["folds"][0]["names"]
    if name not in names0:
        raise KeyError(f"{name} not in {names0}")
    mi = names0.index(name)
    oof_probs, oof_y, oof_subj, fold_accs, fold_meta = [], [], [], [], []
    for k in range(n_folds):
        fit_folds = [bank["folds"][f] for f in range(n_folds) if f != k]
        # 仅温度嵌套（权重恒 1）
        _, probs_fit, y_fit = _concat_member_probs(
            [{**fr, "names": [name], "probs": [fr["probs"][mi]]} for fr in fit_folds]
        )
        cfg = _fit_e1f([name], probs_fit, y_fit)
        pred = fuse_with_config([bank["folds"][k]["probs"][mi]], cfg)
        yk = bank["folds"][k]["y"]
        acc = float(accuracy(pred, yk))
        fold_accs.append(acc)
        oof_probs.append(pred)
        oof_y.append(yk)
        oof_subj.append(bank["folds"][k]["subjects"])
        fold_meta.append({"fold": k, "acc": acc, "config": cfg.to_dict()})
        print(f"  {arm_id} fold{k} acc={acc:.4f}", flush=True)
    return _pack_arm(arm_id, desc, fold_accs, fold_meta, oof_probs, oof_y, oof_subj)


def pooled_e1f_track(bank: dict, arm_id: str, desc: str) -> dict:
    """θ 在全部 6 折 Val 一次拟合，再套到每折（in-sample-θ 诊断）。"""
    n_folds = bank["n_folds"]
    names, probs_all, y_all = _concat_member_probs(bank["folds"])
    cfg = _fit_e1f(names, probs_all, y_all)
    oof_probs, oof_y, oof_subj, fold_accs, fold_meta = [], [], [], [], []
    for k in range(n_folds):
        pred = _apply_e1f(bank["folds"][k], cfg)
        yk = bank["folds"][k]["y"]
        acc = float(accuracy(pred, yk))
        fold_accs.append(acc)
        oof_probs.append(pred)
        oof_y.append(yk)
        oof_subj.append(bank["folds"][k]["subjects"])
        fold_meta.append(
            {
                "fold": k,
                "acc": acc,
                "config": cfg.to_dict(),
                "note": "in-sample-θ (pooled 900)",
            }
        )
        print(f"  {arm_id} fold{k} acc={acc:.4f} (pooled-θ)", flush=True)
    arm = _pack_arm(arm_id, desc, fold_accs, fold_meta, oof_probs, oof_y, oof_subj)
    arm["in_sample_theta"] = True
    arm["pooled_config"] = cfg.to_dict()
    arm["pooled_fit_acc"] = float(accuracy(fuse_with_config(probs_all, cfg), y_all))
    return arm


def nested_stream_probs(bank: dict) -> tuple[list[np.ndarray], list[E1fConfig]]:
    """每折：用其余折拟 E1f，得到该折流概率与 cfg。"""
    n_folds = bank["n_folds"]
    stream_probs = []
    cfgs = []
    for k in range(n_folds):
        fit_folds = [bank["folds"][f] for f in range(n_folds) if f != k]
        names, probs, y_fit = _concat_member_probs(fit_folds)
        cfg = _fit_e1f(names, probs, y_fit)
        stream_probs.append(_apply_e1f(bank["folds"][k], cfg))
        cfgs.append(cfg)
    return stream_probs, cfgs


def nested_uni50(bank_a: dict, bank_b: dict) -> dict:
    """跨流固定 0.5/0.5；流内嵌套 E1f。"""
    n_folds = bank_a["n_folds"]
    sa, cfg_a = nested_stream_probs(bank_a)
    sb, cfg_b = nested_stream_probs(bank_b)
    oof_probs, oof_y, oof_subj, fold_accs, fold_meta = [], [], [], [], []
    for k in range(n_folds):
        pred = (0.5 * sa[k] + 0.5 * sb[k]).astype(np.float32)
        yk = bank_a["folds"][k]["y"]
        acc = float(accuracy(pred, yk))
        fold_accs.append(acc)
        oof_probs.append(pred)
        oof_y.append(yk)
        oof_subj.append(bank_a["folds"][k]["subjects"])
        fold_meta.append(
            {
                "fold": k,
                "acc": acc,
                "stream_e1f_a59": cfg_a[k].to_dict(),
                "stream_e1f_b8": cfg_b[k].to_dict(),
                "stream_weights": {"a59_e1f": 0.5, "b8_ft_e1f": 0.5},
                "w_extra": 0.5,
            }
        )
        print(f"  R-uni50 fold{k} acc={acc:.4f}", flush=True)
    return _pack_arm(
        "R-uni50",
        "fixed 50/50 A59-E1f × B8-E1f (nested streams)",
        fold_accs,
        fold_meta,
        oof_probs,
        oof_y,
        oof_subj,
    )


def _gate(arm: dict, base: dict, *, n_boot: int) -> dict:
    accs = arm["fold_accs"]
    base_accs = base["fold_accs"]
    mean = float(arm["val_acc_mean"])
    base_mean = float(base["val_acc_mean"])
    delta = mean - base_mean
    p_w = wilcoxon_p(accs, base_accs)
    p_t = paired_t_p(accs, base_accs)
    yhat_a = np.concatenate([p.argmax(1) for p in arm["_oof_probs"]])
    yhat_0 = np.concatenate([p.argmax(1) for p in base["_oof_probs"]])
    y = np.concatenate(arm["_oof_y"])
    subj = np.concatenate(arm["_oof_subjects"])
    mcn = mcnemar_exact(yhat_a, yhat_0, y)
    corr_arm = (yhat_a == y).astype(np.float64)
    corr_base = (yhat_0 == y).astype(np.float64)
    boot = cluster_bootstrap_delta(corr_arm, corr_base, subj, n_boot=n_boot)
    return {
        "val_acc_mean": mean,
        "val_acc_std": float(arm["val_acc_std"]),
        "nested_delta_vs_base": delta,
        "base_arm": base["arm_id"],
        "wilcoxon_p": p_w,
        "paired_t_p": p_t,
        "mcnemar": mcn,
        "cluster_bootstrap": {
            "delta": boot["delta"],
            "ci95": boot["ci95"],
            "ci_excludes_zero": boot["ci_excludes_zero"],
            "suspect_single_subject": {
                "flag": boot["suspect_single_subject"]["flag"],
                "top_subject": boot["suspect_single_subject"].get("top_subject"),
                "positive_share": boot["suspect_single_subject"].get("positive_share"),
            },
        },
        "fold_accs": accs,
        "fold_deltas_pp": [float((a - b) * 100) for a, b in zip(accs, base_accs)],
        "in_sample_theta": bool(arm.get("in_sample_theta", False)),
    }


def _uni50_gate(arm: dict, f1: dict) -> dict:
    """§5.3：≥5/6 折 Acc≥F1 同折 且 pooled≥0.53。"""
    a = np.asarray(arm["fold_accs"], dtype=np.float64)
    b = np.asarray(f1["fold_accs"], dtype=np.float64)
    n_ge = int(np.sum(a >= b - 1e-12))
    pooled = float(np.mean(a))  # equal fold sizes
    ok = bool(n_ge >= 5 and pooled >= UNI50_POOLED_LINE - 1e-12)
    return {
        "n_folds_ge_f1": n_ge,
        "pooled_acc": pooled,
        "pooled_line": UNI50_POOLED_LINE,
        "enter_engineering_set": ok,
    }


def _ranking_key(aid: str, arms: dict) -> float:
    """工程排序键：pool 臂用同构 leave-fold（§5.2）。"""
    if aid in ("S0_csv", "N0", "R-pool-S0"):
        return float(arms["N0"]["val_acc_mean"])
    if aid == "R-pool-B8":
        return float(arms["R-B8"]["val_acc_mean"])
    return float(arms[aid]["val_acc_mean"])


def decide_engineering(arms: dict, uni50_ok: bool) -> dict:
    """§5.2 工程选卷。"""
    candidates = ["S0_csv", "R-B8", "V1", "R-pool-B8", "R-pool-S0"]
    if uni50_ok:
        candidates.append("R-uni50")

    scored = []
    for c in candidates:
        key = _ranking_key(c, arms)
        scored.append((c, key, COMPLEXITY.get(c, 99)))
    scored.sort(key=lambda x: (-x[1], x[2], x[0]))
    winner = scored[0][0]
    w_mean = scored[0][1]

    # 平手：差 <1pp 取更简
    tied = [s for s in scored if (w_mean - s[1]) < TIE_PP - 1e-12]
    tied.sort(key=lambda x: (x[2], x[0]))
    final = tied[0][0]

    # 若最终是 pool-S0 且与 S0_csv 同键 → 取已有 S0
    if final == "R-pool-S0":
        final = "S0_csv"
    # pool-B8 与 R-B8 同键且进入平手 → 已由 complexity 偏向 R-B8(3)<pool(4)

    write_new = final not in ("S0_csv", "N0")
    return {
        "candidates": candidates,
        "scored": [
            {"id": c, "ranking_key": k, "complexity": cx} for c, k, cx in scored
        ],
        "winner_raw": winner,
        "decision_engineering": final,
        "write_new_csv": write_new,
        "label": "工程选择，非显著性结论",
        "tie_pool": [t[0] for t in tied],
    }


def decide_science(gates: dict) -> dict:
    """§5.1：科学主行默认 KEEP_S0；Wilcoxon 不作洗白。"""
    rb8 = gates.get("R-B8") or {}
    pw = rb8.get("wilcoxon_p")
    delta = float(rb8.get("nested_delta_vs_base") or 0.0)
    sig = bool(isinstance(pw, float) and pw == pw and pw < 0.05 and delta >= 0.01)
    if sig:
        # 极罕见：显著性过线仍记科学可讨论，但不自动改叙事为「已证明」
        decision = "SCIENCE_RB8_SIGNIFICANT_DISCUSS"
    else:
        decision = "KEEP_S0_discipline"
    return {
        "decision_science": decision,
        "rb8_delta_vs_n0": delta,
        "rb8_wilcoxon_p": pw,
        "note": "默认科学主行维持 S0；Wilcoxon 不作工程换卷唯一闸",
    }


def _pub_arm(arm: dict) -> dict:
    return {k: v for k, v in arm.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer-tag", default=PREFER_TAG)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--reuse-n0-json", action="store_true", help="校验用：仍重算 N0")
    args = ap.parse_args()
    n_boot = int(args.n_boot)

    a59 = find_all_a59_members(prefer_tag=args.prefer_tag)
    b8 = find_all_b8_members(arm="ft", prefer_tag=args.prefer_tag)
    if len(a59) < 4 or len(b8) < 4:
        raise SystemExit(f"成员不足 a59={list(a59)} b8={list(b8)}")
    n_folds = min(n_folds_available(a59), n_folds_available(b8), 6)
    print(f"n_folds={n_folds}", flush=True)

    bank_a = load_track_bank(a59, n_folds)
    bank_b = load_track_bank(b8, n_folds)

    arms: dict[str, dict] = {}

    print("=== N0 (nested-S0) ===", flush=True)
    arms["N0"] = nested_e1f_track(bank_a)
    # alias
    arms["N0"]["arm_id"] = "N0"

    print("=== F1 (a59_conformer) ===", flush=True)
    arms["F1"] = nested_single_member(
        bank_a, "conformer", "F1", "nested single a59_conformer"
    )

    print("=== V1 (b8_shallow_b) ===", flush=True)
    arms["V1"] = nested_single_member(
        bank_b, "shallow_b", "V1", "nested single b8_shallow_b"
    )

    print("=== R-B8 (nested E1f-B8-ft) ===", flush=True)
    arms["R-B8"] = nested_e1f_track(bank_b)
    arms["R-B8"]["arm_id"] = "R-B8"
    arms["R-B8"]["desc"] = "nested E1f-B8-ft (symmetric to N0)"

    print("=== R-pool-S0 ===", flush=True)
    arms["R-pool-S0"] = pooled_e1f_track(
        bank_a, "R-pool-S0", "pooled-θ E1f-A59 (in-sample-θ diagnostic)"
    )

    print("=== R-pool-B8 ===", flush=True)
    arms["R-pool-B8"] = pooled_e1f_track(
        bank_b, "R-pool-B8", "pooled-θ E1f-B8-ft (in-sample-θ diagnostic)"
    )

    print("=== R-uni50 ===", flush=True)
    arms["R-uni50"] = nested_uni50(bank_a, bank_b)

    # 与 Exp37/38 交叉核验
    n0_ref = None
    n0_path = exp37_out() / "replay" / "nested_latest.json"
    if n0_path.is_file():
        n0_ref = json.loads(n0_path.read_text(encoding="utf-8"))
        ref_mean = float((n0_ref.get("arms") or {}).get("N0", {}).get("val_acc_mean", -1))
        print(
            f"cross-check N0: this={arms['N0']['val_acc_mean']:.6f} exp37={ref_mean:.6f}",
            flush=True,
        )

    g38 = exp38_out() / "replay" / "greedy_latest.json"
    if g38.is_file():
        d38 = json.loads(g38.read_text(encoding="utf-8"))
        v1_ref = float((d38.get("arms") or {}).get("V1", {}).get("val_acc_mean", -1))
        print(
            f"cross-check V1: this={arms['V1']['val_acc_mean']:.6f} exp38={v1_ref:.6f}",
            flush=True,
        )

    gates = {
        "N0": {
            "val_acc_mean": arms["N0"]["val_acc_mean"],
            "val_acc_std": arms["N0"]["val_acc_std"],
            "fold_accs": arms["N0"]["fold_accs"],
            "role": "anchor",
        },
        "F1": _gate(arms["F1"], arms["N0"], n_boot=n_boot),
        "V1": _gate(arms["V1"], arms["N0"], n_boot=n_boot),
        "R-B8": _gate(arms["R-B8"], arms["N0"], n_boot=n_boot),
        "R-uni50": _gate(arms["R-uni50"], arms["N0"], n_boot=n_boot),
        "R-pool-S0": _gate(arms["R-pool-S0"], arms["N0"], n_boot=n_boot),
        "R-pool-B8": _gate(arms["R-pool-B8"], arms["N0"], n_boot=n_boot),
    }
    # Q2: R-B8 vs V1
    gates["R-B8_vs_V1"] = _gate(arms["R-B8"], arms["V1"], n_boot=n_boot)
    gates["R-B8_vs_V1"]["base_arm"] = "V1"

    uni50_info = _uni50_gate(arms["R-uni50"], arms["F1"])
    gates["R-uni50"]["uni50_enter"] = uni50_info

    for aid in ("R-B8", "V1", "F1", "R-uni50"):
        g = gates[aid]
        print(
            f"GATE {aid}: nested={g['val_acc_mean']:.4f} "
            f"ΔvsN0={g['nested_delta_vs_base']*100:+.2f}pp W={g['wilcoxon_p']}",
            flush=True,
        )
    print(f"uni50 enter={uni50_info}", flush=True)
    print(
        f"R-B8 vs V1: Δ={gates['R-B8_vs_V1']['nested_delta_vs_base']*100:+.2f}pp "
        f"W={gates['R-B8_vs_V1']['wilcoxon_p']}",
        flush=True,
    )

    sci = decide_science(gates)
    eng = decide_engineering(arms, uni50_ok=uni50_info["enter_engineering_set"])
    print("SCIENCE", sci["decision_science"], flush=True)
    print("ENGINEERING", eng["decision_engineering"], "write_new", eng["write_new_csv"], flush=True)

    out = exp39_out()
    replay = out / "replay"
    preds = out / "preds"
    replay.mkdir(parents=True, exist_ok=True)
    for arm in arms.values():
        save_oof(arm, preds)

    # member dirs for optional CSV
    member_dirs_a59 = {k: str(v) for k, v in a59.items()}
    member_dirs_b8 = {k: str(v) for k, v in b8.items()}

    doc = {
        "experiment": 39,
        "scheme_version": "v0.1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prefer_tag": args.prefer_tag,
        "n_folds": n_folds,
        "n_boot": n_boot,
        "member_dirs_a59": member_dirs_a59,
        "member_dirs_b8": member_dirs_b8,
        "arms": {k: _pub_arm(v) for k, v in arms.items()},
        "gates": gates,
        "uni50_enter": uni50_info,
        "decision_science": sci,
        "decision_engineering": eng,
        "honest_ranking": sorted(
            [
                {
                    "arm": aid,
                    "nested_mean": float(arms[aid]["val_acc_mean"]),
                    "nested_std": float(arms[aid]["val_acc_std"]),
                    "in_sample_theta": bool(arms[aid].get("in_sample_theta", False)),
                }
                for aid in ("V1", "R-B8", "R-uni50", "N0", "F1")
            ],
            key=lambda r: -r["nested_mean"],
        ),
        "note": (
            "Honest ranking excludes R-pool-* (in-sample-θ). "
            "Science KEEP_S0_discipline vs engineering mean+simplicity dual track."
        ),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (replay / f"ranking_{stamp}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (replay / "ranking_latest.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "metrics_full_latest.json").write_text(
        json.dumps(
            {
                "experiment": 39,
                "generated_at": doc["generated_at"],
                "decision_science": sci["decision_science"],
                "decision_engineering": eng["decision_engineering"],
                "write_new_csv": eng["write_new_csv"],
                "honest_ranking": doc["honest_ranking"],
                "R-B8": {
                    "nested": gates["R-B8"]["val_acc_mean"],
                    "delta_vs_n0": gates["R-B8"]["nested_delta_vs_base"],
                    "wilcoxon_p": gates["R-B8"]["wilcoxon_p"],
                    "delta_vs_v1": gates["R-B8_vs_V1"]["nested_delta_vs_base"],
                },
                "uni50_enter": uni50_info,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", replay / "ranking_latest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
