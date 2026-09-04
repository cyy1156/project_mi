# -*- coding: utf-8 -*-
"""Day0：M7 跨轨融合 + MC0 类边际校正（零训练回放）。

用法：
  python replay_m7_margin.py
  python replay_m7_margin.py --skip-m7c
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_STEP = Path(__file__).resolve().parent
from exp36_config import (  # noqa: E402
    ANCHOR_S0,
    PREFER_TAG,
    REPLACE_LINE,
    W_B8_MAX,
    a59_step,
    exp36_out,
    rankflip_step,
)

_RF = rankflip_step()
_A59 = a59_step()
for p in (_RF, _A59, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e1f_constrained import fit_e1f_constrained, fuse_with_config  # noqa: E402
from e1f_core import E1fConfig, apply_temperature, accuracy  # noqa: E402
from exp35_config import FusionConstraints, MEMBER_KEYS  # noqa: E402
from member_paths import find_all_a59_members, find_all_b8_members, n_folds_available  # noqa: E402
from metrics_three import three_class_report  # noqa: E402


def _load_pool_probs(member_dirs: dict[str, Path], fold: int, pool: list[str]):
    names, probs = [], []
    y_ref = idx_ref = None
    for name in pool:
        fd = member_dirs[name] / f"fold{fold}"
        p = np.load(fd / "val_prob.npy")
        y = np.load(fd / "val_y.npy")
        idx = np.load(fd / "val_idx.npy") if (fd / "val_idx.npy").is_file() else np.arange(len(y))
        if y_ref is None:
            y_ref, idx_ref = y, idx
        else:
            if not np.array_equal(y, y_ref):
                raise RuntimeError(f"fold{fold} {name} val_y mismatch")
            if not np.array_equal(idx, idx_ref):
                raise RuntimeError(f"fold{fold} {name} val_idx mismatch")
        names.append(name)
        probs.append(p)
    assert y_ref is not None and idx_ref is not None
    return names, probs, y_ref, idx_ref


def _fuse_track(member_dirs: dict[str, Path], fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    names, probs, y, idx = _load_pool_probs(member_dirs, fold, list(MEMBER_KEYS))
    cons = FusionConstraints(name="unconstrained")
    cfg = fit_e1f_constrained(names, probs, y, cons)
    fused = fuse_with_config(probs, cfg)
    return fused, y, idx, cfg.to_dict()


def _fit_two_stream(
    p_a: np.ndarray,
    p_b: np.ndarray,
    y: np.ndarray,
    *,
    w_b_max: float = W_B8_MAX,
) -> tuple[E1fConfig, np.ndarray]:
    """两流：names=a59_e1f, b8_ft_e1f；限制 w_b ≤ w_b_max。"""
    names = ["a59_e1f", "b8_ft_e1f"]
    cons = FusionConstraints(name="M7_w_b8_cap")
    # 借用 eegnet 槽位名不可用；手写网格
    from e1f_core import fit_temperature

    t_a = fit_temperature(p_a, y)
    t_b = fit_temperature(p_b, y)
    ca = apply_temperature(p_a, t_a)
    cb = apply_temperature(p_b, t_b)
    best_w, best_acc = 0.0, -1.0
    for w_b in np.arange(0.0, w_b_max + 1e-9, 0.05):
        w_a = 1.0 - float(w_b)
        fused = w_a * ca + float(w_b) * cb
        acc = accuracy(fused, y)
        if acc > best_acc + 1e-12:
            best_acc = acc
            best_w = float(w_b)
    w_a = 1.0 - best_w
    fused = w_a * ca + best_w * cb
    cfg = E1fConfig(
        member_names=names,
        temperatures=[float(t_a), float(t_b)],
        weights=[float(w_a), float(best_w)],
        smooth_radius=0,
        val_acc=float(best_acc),
    )
    return cfg, fused


def _apply_bias(probs: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    logits = np.log(p) + b.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _fit_bias(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.astype(int)

    def nll(b: np.ndarray) -> float:
        q = _apply_bias(probs, b)
        q = np.clip(q, 1e-12, 1.0)
        return float(-np.log(q[np.arange(len(y)), y]).mean())

    res = minimize(nll, np.zeros(3), method="L-BFGS-B")
    return np.asarray(res.x, dtype=np.float64)


def run_s0_and_streams(prefer_tag: str) -> dict:
    a59 = find_all_a59_members(prefer_tag=prefer_tag)
    b8 = find_all_b8_members(arm="ft", prefer_tag=prefer_tag)
    if len(a59) < 4 or len(b8) < 4:
        raise SystemExit(f"成员不足 a59={list(a59)} b8={list(b8)}")
    n_folds = min(n_folds_available(a59), n_folds_available(b8))
    folds = []
    for fold in range(n_folds):
        pa, ya, ia, cfg_a = _fuse_track(a59, fold)
        pb, yb, ib, cfg_b = _fuse_track(b8, fold)
        if not np.array_equal(ya, yb) or not np.array_equal(ia, ib):
            raise RuntimeError(f"fold{fold} A59/B8 对齐失败")
        folds.append(
            {
                "fold": fold,
                "y": ya,
                "idx": ia,
                "p_a59": pa,
                "p_b8": pb,
                "cfg_a59": cfg_a,
                "cfg_b8": cfg_b,
            }
        )
    return {"n_folds": n_folds, "folds": folds, "a59_dirs": {k: str(v) for k, v in a59.items()}, "b8_dirs": {k: str(v) for k, v in b8.items()}}


def arm_s0(streams: dict) -> dict:
    fold_recs = []
    for fr in streams["folds"]:
        report = three_class_report(fr["y"], fr["p_a59"].argmax(axis=1))
        fold_recs.append(
            {
                "fold": fr["fold"],
                "val_metrics": report,
                "config": fr["cfg_a59"],
                "probs": fr["p_a59"],
                "y": fr["y"],
            }
        )
    accs = [float(r["val_metrics"]["acc"]) for r in fold_recs]
    return {
        "arm_id": "S0",
        "desc": "E1f-A59 无约束（复现锚）",
        "n_fit_params": 0,
        "val_acc_mean": float(np.mean(accs)),
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "fold_accs": accs,
        "folds": [
            {k: v for k, v in r.items() if k not in ("probs", "y")} | {"n": int(len(r["y"]))}
            for r in fold_recs
        ],
        "_fold_probs": [r["probs"] for r in fold_recs],
        "_fold_y": [r["y"] for r in fold_recs],
    }


def arm_m7(streams: dict) -> dict:
    fold_recs = []
    for fr in streams["folds"]:
        cfg, fused = _fit_two_stream(fr["p_a59"], fr["p_b8"], fr["y"], w_b_max=W_B8_MAX)
        report = three_class_report(fr["y"], fused.argmax(axis=1))
        fold_recs.append(
            {
                "fold": fr["fold"],
                "val_metrics": report,
                "config": cfg.to_dict(),
                "w_b8": float(cfg.weights[1]),
                "probs": fused,
                "y": fr["y"],
            }
        )
    accs = [float(r["val_metrics"]["acc"]) for r in fold_recs]
    return {
        "arm_id": "M7",
        "desc": f"A59-E1f × B8-ft-E1f · w_B8≤{W_B8_MAX}",
        "n_fit_params": "≤3 per fold (2×T + w)",
        "val_acc_mean": float(np.mean(accs)),
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "fold_accs": accs,
        "w_b8_mean": float(np.mean([r["w_b8"] for r in fold_recs])),
        "w_b8_max": float(np.max([r["w_b8"] for r in fold_recs])),
        "folds": [
            {k: v for k, v in r.items() if k not in ("probs", "y")} | {"n": int(len(r["y"]))}
            for r in fold_recs
        ],
        "_fold_probs": [r["probs"] for r in fold_recs],
        "_fold_y": [r["y"] for r in fold_recs],
    }


def arm_mc0(s0: dict) -> dict:
    """nested：留一折拟 b；pooled：全 900 拟 b（附报）。"""
    probs_l = s0["_fold_probs"]
    y_l = s0["_fold_y"]
    n_folds = len(probs_l)

    # nested
    nested_accs = []
    nested_folds = []
    for i in range(n_folds):
        p_tr = np.concatenate([probs_l[j] for j in range(n_folds) if j != i], axis=0)
        y_tr = np.concatenate([y_l[j] for j in range(n_folds) if j != i], axis=0)
        b = _fit_bias(p_tr, y_tr)
        p_te = _apply_bias(probs_l[i], b)
        report = three_class_report(y_l[i], p_te.argmax(axis=1))
        nested_accs.append(float(report["acc"]))
        nested_folds.append(
            {
                "fold": i,
                "bias": b.tolist(),
                "val_metrics": report,
            }
        )

    # pooled
    p_all = np.concatenate(probs_l, axis=0)
    y_all = np.concatenate(y_l, axis=0)
    b_pool = _fit_bias(p_all, y_all)
    pooled_fold_accs = []
    for i in range(n_folds):
        p_i = _apply_bias(probs_l[i], b_pool)
        pooled_fold_accs.append(float(three_class_report(y_l[i], p_i.argmax(axis=1))["acc"]))
    pooled_acc = float(accuracy(p_all, y_all))  # optimistic single number

    return {
        "arm_id": "MC0",
        "desc": "类边际校正 on S0（主=nested）",
        "n_fit_params": 3,
        "nested": {
            "val_acc_mean": float(np.mean(nested_accs)),
            "val_acc_std": float(np.std(nested_accs, ddof=1)) if len(nested_accs) > 1 else 0.0,
            "fold_accs": nested_accs,
            "folds": nested_folds,
        },
        "pooled_appendix": {
            "bias": b_pool.tolist(),
            "pooled_acc_all900": pooled_acc,
            "fold_accs_with_global_b": pooled_fold_accs,
            "val_acc_mean": float(np.mean(pooled_fold_accs)),
            "val_acc_std": float(np.std(pooled_fold_accs, ddof=1)) if len(pooled_fold_accs) > 1 else 0.0,
            "note": "乐观附报；主决策用 nested",
        },
        "fold_accs": nested_accs,
        "val_acc_mean": float(np.mean(nested_accs)),
        "val_acc_std": float(np.std(nested_accs, ddof=1)) if len(nested_accs) > 1 else 0.0,
    }


def arm_m7c(m7: dict, streams: dict) -> dict:
    """M7 概率上 nested 边际校正。"""
    probs_l = m7["_fold_probs"]
    y_l = m7["_fold_y"]
    n_folds = len(probs_l)
    nested_accs = []
    nested_folds = []
    for i in range(n_folds):
        p_tr = np.concatenate([probs_l[j] for j in range(n_folds) if j != i], axis=0)
        y_tr = np.concatenate([y_l[j] for j in range(n_folds) if j != i], axis=0)
        b = _fit_bias(p_tr, y_tr)
        p_te = _apply_bias(probs_l[i], b)
        report = three_class_report(y_l[i], p_te.argmax(axis=1))
        nested_accs.append(float(report["acc"]))
        nested_folds.append({"fold": i, "bias": b.tolist(), "val_metrics": report})
    return {
        "arm_id": "M7c",
        "desc": "M7 + nested 边际校正",
        "n_fit_params": "M7 + 3",
        "val_acc_mean": float(np.mean(nested_accs)),
        "val_acc_std": float(np.std(nested_accs, ddof=1)) if len(nested_accs) > 1 else 0.0,
        "fold_accs": nested_accs,
        "folds": nested_folds,
    }


def _wilcoxon(a: list[float], b: list[float]) -> float:
    diff = [x - y for x, y in zip(a, b)]
    d = [x for x in diff if x != 0.0]
    if len(d) < 1:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _gate(arm: dict, s0: dict) -> dict:
    acc = float(arm["val_acc_mean"])
    p = _wilcoxon(arm["fold_accs"], s0["fold_accs"])
    plus = acc >= REPLACE_LINE
    sig = p == p and p < 0.05
    return {
        "val_acc": acc,
        "delta_vs_s0": acc - float(s0["val_acc_mean"]),
        "wilcoxon_p": p,
        "replace_line": REPLACE_LINE,
        "pass_val": plus,
        "pass_sig": bool(sig),
        "pass_replace": bool(plus and sig),
    }


def _strip_private(arm: dict) -> dict:
    return {k: v for k, v in arm.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer-tag", default=PREFER_TAG)
    ap.add_argument("--skip-m7c", action="store_true")
    args = ap.parse_args()

    streams = run_s0_and_streams(args.prefer_tag)
    s0 = arm_s0(streams)
    m7 = arm_m7(streams)
    mc0 = arm_mc0(s0)
    arms = {"S0": s0, "M7": m7, "MC0": mc0}
    if not args.skip_m7c:
        arms["M7c"] = arm_m7c(m7, streams)

    gates = {k: _gate(v, s0) for k, v in arms.items() if k != "S0"}
    # MC0 gate uses nested fold_accs already on arm
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp36_out() / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scheme": "Exp36 v0.1 Day0",
        "anchor_s0_expected": ANCHOR_S0,
        "prefer_tag": args.prefer_tag,
        "alignment": "A59/B8 val_y & val_idx asserted equal per fold",
        "arms": {k: _strip_private(v) for k, v in arms.items()},
        "gates": gates,
        "paths": {"a59": streams["a59_dirs"], "b8_ft": streams["b8_dirs"]},
    }
    out = out_dir / f"day0_{stamp}.json"
    latest = out_dir / "day0_latest.json"
    # JSON: drop huge if any — already stripped probs
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    out.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")

    print(f"S0  {s0['val_acc_mean']:.4f}±{s0['val_acc_std']:.4f}  (anchor≈{ANCHOR_S0})")
    for k, g in gates.items():
        print(
            f"{k:4s} {g['val_acc']:.4f}  Δ={g['delta_vs_s0']:+.4f}  "
            f"p={g['wilcoxon_p']:.4f}  pass={g['pass_replace']}"
        )
    print("wrote", out)
    print("wrote", latest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
