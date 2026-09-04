# -*- coding: utf-8 -*-
"""Exp36 Day1 回放：P1 扩池 E1f + M7b（B8 多种子均值）+ 单模 B3 对照。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent
_RF = _STEP.parent / "5070_challenge_rankflip_accpaper"
_A59 = _STEP.parent / "5070_challenge_mi_59ch_accpaper"
for p in (_RF, _A59, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e1f_constrained import fit_e1f_constrained, fuse_with_config  # noqa: E402
from e1f_core import E1fConfig, apply_temperature, accuracy, fit_temperature  # noqa: E402
from exp35_config import FusionConstraints, MEMBER_KEYS  # noqa: E402
from exp36_config import ANCHOR_S0, PREFER_TAG, REPLACE_LINE, W_B8_MAX, exp36_out  # noqa: E402
from member_paths import find_all_a59_members, find_all_b8_members, n_folds_available  # noqa: E402
from metrics_three import three_class_report  # noqa: E402
from replay_m7_margin import _fit_bias, _apply_bias, _wilcoxon, _fit_two_stream  # noqa: E402


def _find_exp36_three(track: str, arm: str, folder: str) -> Path | None:
    root = exp36_out() / track / arm / folder
    if not root.is_dir():
        return None
    runs = sorted(root.glob("run_*/three"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _load_probs(three: Path, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fd = three / f"fold{fold}"
    p = np.load(fd / "val_prob.npy")
    y = np.load(fd / "val_y.npy")
    idx = np.load(fd / "val_idx.npy") if (fd / "val_idx.npy").is_file() else np.arange(len(y))
    return p, y, idx


def _fit_pool(member_dirs: dict[str, Path], n_folds: int, arm_id: str, desc: str) -> dict:
    fold_recs = []
    fold_accs = []
    for fold in range(n_folds):
        names, probs, y_ref = [], [], None
        for name, three in member_dirs.items():
            p, y, _ = _load_probs(three, fold)
            if y_ref is None:
                y_ref = y
            elif not np.array_equal(y, y_ref):
                raise RuntimeError(f"{arm_id} fold{fold} y mismatch {name}")
            names.append(name)
            probs.append(p)
        assert y_ref is not None
        # 大池用更粗网格：改 step via 临时约束名 + 手写连续优化
        if len(names) <= 5:
            cons = FusionConstraints(name=f"{arm_id}_unc")
            cfg = fit_e1f_constrained(names, probs, y_ref, cons)
            fused = fuse_with_config(probs, cfg)
            cfg_d = cfg.to_dict()
        else:
            cfg_d, fused = _fit_continuous(names, probs, y_ref)
        report = three_class_report(y_ref, fused.argmax(axis=1))
        fold_accs.append(float(report["acc"]))
        fold_recs.append({"fold": fold, "config": cfg_d, "val_metrics": report})
    return {
        "arm_id": arm_id,
        "desc": desc,
        "pool": list(member_dirs.keys()),
        "n_fit_params": f"T×{len(member_dirs)}+w simplex",
        "val_acc_mean": float(np.mean(fold_accs)),
        "val_acc_std": float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0,
        "fold_accs": fold_accs,
        "folds": fold_recs,
    }


def _fit_continuous(names: list[str], probs: list[np.ndarray], y: np.ndarray) -> tuple[dict, np.ndarray]:
    from scipy.optimize import minimize

    temps = [fit_temperature(p, y) for p in probs]
    cal = [apply_temperature(p, t) for p, t in zip(probs, temps)]
    stack = np.stack(cal, axis=0)  # (M,N,C)
    m = len(names)

    def nll(z: np.ndarray) -> float:
        w = np.exp(z - z.max())
        w = w / w.sum()
        fused = np.tensordot(w, stack, axes=(0, 0))
        q = np.clip(fused, 1e-12, 1.0)
        return float(-np.log(q[np.arange(len(y)), y.astype(int)]).mean())

    res = minimize(nll, np.zeros(m), method="L-BFGS-B")
    z = res.x
    w = np.exp(z - z.max())
    w = w / w.sum()
    fused = np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)
    cfg = {
        "member_names": names,
        "temperatures": [float(t) for t in temps],
        "weights": [float(x) for x in w],
        "smooth_radius": 0,
        "val_acc": float(accuracy(fused, y)),
        "fit": "continuous_softmax",
    }
    return cfg, fused


def _gate(arm: dict, s0_accs: list[float]) -> dict:
    acc = float(arm["val_acc_mean"])
    p = _wilcoxon(arm["fold_accs"], s0_accs)
    return {
        "val_acc": acc,
        "delta_vs_s0": acc - float(np.mean(s0_accs)),
        "wilcoxon_p": p,
        "pass_val": acc >= REPLACE_LINE,
        "pass_sig": bool(p == p and p < 0.05),
        "pass_replace": bool(acc >= REPLACE_LINE and p == p and p < 0.05),
    }


def _mean_probs(threes: list[Path], fold: int) -> tuple[np.ndarray, np.ndarray]:
    ps, y0 = [], None
    for th in threes:
        p, y, _ = _load_probs(th, fold)
        if y0 is None:
            y0 = y
        elif not np.array_equal(y, y0):
            raise RuntimeError("B8 seed y mismatch")
        ps.append(p)
    assert y0 is not None
    return np.mean(np.stack(ps, axis=0), axis=0).astype(np.float32), y0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer-tag", default=PREFER_TAG)
    args = ap.parse_args()

    base_a59 = find_all_a59_members(prefer_tag=args.prefer_tag)
    base_b8 = find_all_b8_members(arm="ft", prefer_tag=args.prefer_tag)
    n_folds = n_folds_available(base_a59)

    # S0 fold accs from day0 if present
    day0 = exp36_out() / "replay" / "day0_latest.json"
    if day0.is_file():
        s0_accs = json.loads(day0.read_text(encoding="utf-8"))["arms"]["S0"]["fold_accs"]
    else:
        # recompute quickly
        from replay_m7_margin import run_s0_and_streams, arm_s0

        s0_accs = arm_s0(run_s0_and_streams(args.prefer_tag))["fold_accs"]

    arms: dict[str, dict] = {}

    # P0: original 4 (sanity)
    arms["P0"] = _fit_pool(base_a59, n_folds, "P0", "Exp34 四成员复现")

    # P1: + conformer s43/s44 + shallow/eeg s43
    pool = dict(base_a59)
    for folder, key in [
        ("conformer_s43", "conformer_s43"),
        ("conformer_s44", "conformer_s44"),
        ("shallow_s43", "shallow_s43"),
        ("eegnet_s43", "eegnet_s43"),
    ]:
        th = _find_exp36_three("A59", "B1" if "conformer_s" in folder else "B2", folder)
        if th is None and "conformer_s" in folder:
            th = _find_exp36_three("A59", "B1", folder)
        if th is not None:
            pool[key] = th
    arms["P1"] = _fit_pool(pool, n_folds, "P1", "四成员+多种子扩池")

    # P1b: only add best B3 variant if exists
    pool_b = dict(base_a59)
    for folder, key in [
        ("conformer_pat40_s42", "conformer_pat40"),
        ("conformer_drop025_s42", "conformer_drop025"),
        ("conformer_s43", "conformer_s43"),
        ("conformer_s44", "conformer_s44"),
    ]:
        arm_dir = "B3" if "pat40" in folder or "drop025" in folder else "B1"
        th = _find_exp36_three("A59", arm_dir, folder)
        if th is not None:
            pool_b[key] = th
    if len(pool_b) > 4:
        arms["P1b"] = _fit_pool(pool_b, n_folds, "P1b", "四成员+B1/B3 变体")

    # B3 single-model report
    for folder, aid in [("conformer_pat40_s42", "B3_pat40"), ("conformer_drop025_s42", "B3_drop025")]:
        th = _find_exp36_three("A59", "B3", folder)
        if th is None:
            continue
        accs = []
        for fold in range(n_folds):
            p, y, _ = _load_probs(th, fold)
            accs.append(float(three_class_report(y, p.argmax(1))["acc"]))
        arms[aid] = {
            "arm_id": aid,
            "desc": folder,
            "val_acc_mean": float(np.mean(accs)),
            "val_acc_std": float(np.std(accs, ddof=1)),
            "fold_accs": accs,
            "n_fit_params": 0,
        }

    # M7b: A59 E1f × B8-ft 多种子均值 E1f
    # Build per-fold p_a59 and p_b8_avg
    fold_recs = []
    fold_accs = []
    for fold in range(n_folds):
        # A59 stream
        names, probs, y_ref = [], [], None
        for name, three in base_a59.items():
            p, y, _ = _load_probs(three, fold)
            if y_ref is None:
                y_ref = y
            names.append(name)
            probs.append(p)
        assert y_ref is not None
        cfg_a = fit_e1f_constrained(names, probs, y_ref, FusionConstraints(name="a59"))
        p_a = fuse_with_config(probs, cfg_a)

        # B8: for each member, mean over available seeds (42 base + exp36)
        b8_probs = []
        b8_names = []
        for name in MEMBER_KEYS:
            threes = []
            if name in base_b8:
                threes.append(base_b8[name])
            for folder in (f"{name}_s43", f"{name}_s44"):
                th = _find_exp36_three("B8ft", "B4", folder)
                if th is not None:
                    threes.append(th)
            if not threes:
                continue
            pm, yb = _mean_probs(threes, fold)
            if not np.array_equal(yb, y_ref):
                raise RuntimeError("B8/A59 y mismatch")
            b8_names.append(name)
            b8_probs.append(pm)
        cfg_b = fit_e1f_constrained(b8_names, b8_probs, y_ref, FusionConstraints(name="b8avg"))
        p_b = fuse_with_config(b8_probs, cfg_b)
        cfg_m, fused = _fit_two_stream(p_a, p_b, y_ref, w_b_max=W_B8_MAX)
        report = three_class_report(y_ref, fused.argmax(1))
        fold_accs.append(float(report["acc"]))
        fold_recs.append(
            {
                "fold": fold,
                "config": cfg_m.to_dict(),
                "w_b8": float(cfg_m.weights[1]),
                "val_metrics": report,
                "n_b8_members": len(b8_names),
            }
        )
    arms["M7b"] = {
        "arm_id": "M7b",
        "desc": "M7 + B8-ft 多种子均值",
        "n_fit_params": "≤3/fold",
        "val_acc_mean": float(np.mean(fold_accs)),
        "val_acc_std": float(np.std(fold_accs, ddof=1)),
        "fold_accs": fold_accs,
        "w_b8_mean": float(np.mean([r["w_b8"] for r in fold_recs])),
        "folds": fold_recs,
    }

    gates = {k: _gate(v, s0_accs) for k, v in arms.items()}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp36_out() / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scheme": "Exp36 Day1",
        "anchor_s0": float(np.mean(s0_accs)),
        "s0_fold_accs": s0_accs,
        "replace_line": REPLACE_LINE,
        "arms": arms,
        "gates": gates,
        "any_pass_replace": any(g.get("pass_replace") for g in gates.values()),
    }
    out = out_dir / f"day1_{stamp}.json"
    latest = out_dir / "day1_latest.json"
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    out.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")

    print(f"S0 mean {np.mean(s0_accs):.4f}")
    for k, g in gates.items():
        print(
            f"{k:10s} {g['val_acc']:.4f} Δ={g['delta_vs_s0']:+.4f} "
            f"p={g['wilcoxon_p']} pass={g['pass_replace']}"
        )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
