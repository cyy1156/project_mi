# -*- coding: utf-8 -*-
"""Exp38 D2：嵌套贪心选池 + 统一入池对照（U0/F1/V1）。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent
from exp38_config import (  # noqa: E402
    CLASSICAL_CANDIDATES,
    DELTA_LINE,
    GREEDY_STOP_PP,
    MEMBER_KEYS,
    NEURAL_CANDIDATES,
    OUT_ROOT_TAG,
    POOL_MAX,
    PREFER_TAG,
    RUN_TAG,
    W_B8_MAX,
    W_C1_MAX,
    a59_step,
    exp36_out,
    exp37_out,
    exp37_step,
    exp38_out,
    rankflip_step,
    scheme_doc,
)

_RF = rankflip_step()
_A59 = a59_step()
_E37 = exp37_step()
for p in (_RF, _A59, _E37, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e1f_constrained import fit_e1f_constrained, fuse_with_config  # noqa: E402
from e1f_core import accuracy  # noqa: E402
from exp35_config import FusionConstraints  # noqa: E402
from member_paths import find_all_a59_members, find_all_b8_members  # noqa: E402
from metrics_secondary import paired_t_p, wilcoxon_p  # noqa: E402


def _load_fold(three: Path, fold: int):
    fd = three / f"fold{fold}"
    p = np.load(fd / "val_prob.npy").astype(np.float32)
    y = np.load(fd / "val_y.npy").astype(np.int64)
    idx = np.load(fd / "val_idx.npy") if (fd / "val_idx.npy").is_file() else np.arange(len(y))
    return p, y, idx.astype(np.int64)


def _find_latest_three(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    runs = sorted(root.glob("run_*/three"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _find_c1() -> Path:
    cands = sorted(
        (exp36_out() / "C1").glob("ft_conformer_*/challenge_mi_3s_45ch/run_*/three"),
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError("C1 missing")
    return cands[-1]


def _find_family(name: str, run_tag: str) -> Path | None:
    root = exp38_out() / f"{name}_challenge_mi_3s_59ch" / "challenge_mi_3s_59ch"
    # prefer exact run_tag
    exact = root / f"run_{run_tag}" / "three"
    if exact.is_dir() and (exact / "fold0" / "val_prob.npy").is_file():
        return exact
    return _find_latest_three(root)


def collect_candidates(run_tag: str, prefer_tag: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    a59 = find_all_a59_members(prefer_tag=prefer_tag)
    b8 = find_all_b8_members(arm="ft", prefer_tag=prefer_tag)
    for k, p in a59.items():
        out[f"a59_{k}"] = p
    for k, p in b8.items():
        out[f"b8_{k}"] = p
    out["c1_conformer"] = _find_c1()
    for name in list(NEURAL_CANDIDATES) + list(CLASSICAL_CANDIDATES):
        th = _find_family(name, run_tag)
        if th is not None:
            out[f"fam_{name}"] = th
        else:
            print(f"WARN missing family {name}", flush=True)
    return out


def load_all_folds(cands: dict[str, Path], n_folds: int = 6) -> dict:
    """cand -> list[fold] of (prob, y, idx)."""
    bank = {}
    y_ref = idx_ref = None
    for name, three in cands.items():
        folds = []
        for f in range(n_folds):
            p, y, idx = _load_fold(three, f)
            if y_ref is None:
                y_ref, idx_ref = [None] * n_folds, [None] * n_folds
            if y_ref[f] is None:
                y_ref[f], idx_ref[f] = y, idx
            elif not np.array_equal(y, y_ref[f]) or not np.array_equal(idx, idx_ref[f]):
                raise RuntimeError(f"{name} fold{f} align fail")
            folds.append({"prob": p, "y": y, "idx": idx})
        bank[name] = folds
    return {"bank": bank, "y_ref": y_ref, "n_folds": n_folds, "names": list(cands.keys())}


def nested_e1f_score(bank: dict, pool: list[str]) -> tuple[float, list[float], list[dict]]:
    """Leave-fold 嵌套 E1f Acc mean。"""
    n_folds = bank["n_folds"]
    fold_accs = []
    fold_meta = []
    for k in range(n_folds):
        fit_idx = [f for f in range(n_folds) if f != k]
        names = list(pool)
        probs_fit = [
            np.concatenate([bank["bank"][n][f]["prob"] for f in fit_idx], axis=0) for n in names
        ]
        y_fit = np.concatenate([bank["y_ref"][f] for f in fit_idx], axis=0)
        cfg = fit_e1f_constrained(names, probs_fit, y_fit, FusionConstraints(name="nested"))
        probs_k = [bank["bank"][n][k]["prob"] for n in names]
        pred = fuse_with_config(probs_k, cfg)
        yk = bank["y_ref"][k]
        acc = float(accuracy(pred, yk))
        fold_accs.append(acc)
        fold_meta.append({"fold": k, "acc": acc, "config": cfg.to_dict()})
    return float(np.mean(fold_accs)), fold_accs, fold_meta


def nested_single_acc(bank: dict, name: str) -> float:
    mean, _, _ = nested_e1f_score(bank, [name])
    return mean


def greedy_select(bank: dict, candidates: list[str]) -> dict:
    # seed = best nested single
    singles = {n: nested_single_acc(bank, n) for n in candidates}
    seed = max(singles, key=singles.get)
    pool = [seed]
    path = [{"step": 0, "add": seed, "nested_acc": singles[seed], "gain_pp": None}]
    print(f"greedy seed={seed} acc={singles[seed]:.4f}", flush=True)

    while len(pool) < POOL_MAX:
        best_c, best_gain, best_acc = None, -1.0, None
        base_acc, _, _ = nested_e1f_score(bank, pool)
        for c in candidates:
            if c in pool:
                continue
            acc, _, _ = nested_e1f_score(bank, pool + [c])
            gain = acc - base_acc
            if gain > best_gain + 1e-12:
                best_gain, best_c, best_acc = gain, c, acc
        if best_c is None or best_gain < GREEDY_STOP_PP - 1e-12:
            print(f"stop gain={best_gain} cand={best_c}", flush=True)
            break
        pool.append(best_c)
        path.append(
            {
                "step": len(pool) - 1,
                "add": best_c,
                "nested_acc": best_acc,
                "gain_pp": float(best_gain * 100),
            }
        )
        print(f"  +{best_c} → {best_acc:.4f} (Δ{best_gain*100:+.2f}pp) pool={pool}", flush=True)

    final_mean, final_accs, final_meta = nested_e1f_score(bank, pool)
    return {
        "pool": pool,
        "path": path,
        "singles": singles,
        "val_acc_mean": final_mean,
        "val_acc_std": float(np.std(final_accs, ddof=1)) if len(final_accs) > 1 else 0.0,
        "fold_accs": final_accs,
        "folds": final_meta,
    }


def nested_multistream_u0(bank: dict) -> dict:
    """对照 U0：两级 A59-E1f × B8-E1f（嵌套），同 Exp37 N7 精神。"""
    # reuse stream nesting lightly
    a59_names = [f"a59_{k}" for k in MEMBER_KEYS if f"a59_{k}" in bank["bank"]]
    b8_names = [f"b8_{k}" for k in MEMBER_KEYS if f"b8_{k}" in bank["bank"]]
    n_folds = bank["n_folds"]
    fold_accs = []
    for k in range(n_folds):
        fit_idx = [f for f in range(n_folds) if f != k]

        def fit_stream(names):
            probs = [
                np.concatenate([bank["bank"][n][f]["prob"] for f in fit_idx], axis=0) for n in names
            ]
            y = np.concatenate([bank["y_ref"][f] for f in fit_idx], axis=0)
            return fit_e1f_constrained(names, probs, y, FusionConstraints(name="s"))

        cfg_a = fit_stream(a59_names)
        cfg_b = fit_stream(b8_names)
        # stream probs on fit
        from e1f_core import apply_temperature, fit_temperature

        def stream_on_folds(cfg, names, folds):
            outs = []
            for f in folds:
                probs = [bank["bank"][n][f]["prob"] for n in names]
                outs.append(fuse_with_config(probs, cfg))
            return outs

        pa_fit = np.concatenate(stream_on_folds(cfg_a, a59_names, fit_idx), axis=0)
        pb_fit = np.concatenate(stream_on_folds(cfg_b, b8_names, fit_idx), axis=0)
        y_fit = np.concatenate([bank["y_ref"][f] for f in fit_idx], axis=0)
        ta, tb = fit_temperature(pa_fit, y_fit), fit_temperature(pb_fit, y_fit)
        ca, cb = apply_temperature(pa_fit, ta), apply_temperature(pb_fit, tb)
        best_w, best_acc = 0.0, -1.0
        for w_b in np.arange(0.0, W_B8_MAX + 1e-9, 0.05):
            fused = (1 - w_b) * ca + w_b * cb
            acc = accuracy(fused, y_fit)
            if acc > best_acc:
                best_acc, best_w = acc, float(w_b)
        pa_k = stream_on_folds(cfg_a, a59_names, [k])[0]
        pb_k = stream_on_folds(cfg_b, b8_names, [k])[0]
        pred = (1 - best_w) * apply_temperature(pa_k, ta) + best_w * apply_temperature(pb_k, tb)
        fold_accs.append(float(accuracy(pred, bank["y_ref"][k])))
    return {
        "arm_id": "U0",
        "val_acc_mean": float(np.mean(fold_accs)),
        "val_acc_std": float(np.std(fold_accs, ddof=1)),
        "fold_accs": fold_accs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default=RUN_TAG)
    ap.add_argument("--prefer-tag", default=PREFER_TAG)
    args = ap.parse_args()

    cands = collect_candidates(args.run_tag, args.prefer_tag)
    print("candidates", list(cands.keys()), flush=True)
    bank = load_all_folds(cands, n_folds=6)

    # A0 = nested original A59 four
    a0_pool = [f"a59_{k}" for k in MEMBER_KEYS]
    a0_mean, a0_accs, _ = nested_e1f_score(bank, a0_pool)
    a0 = {
        "arm_id": "A0",
        "pool": a0_pool,
        "val_acc_mean": a0_mean,
        "val_acc_std": float(np.std(a0_accs, ddof=1)),
        "fold_accs": a0_accs,
        "note": "nested-S0 / Exp37 N0 同构（A59 四成员嵌套 E1f）",
    }
    print(f"A0={a0_mean:.4f}", flush=True)

    # G*: greedy on all available
    print("=== G* greedy all ===", flush=True)
    gstar = greedy_select(bank, list(cands.keys()))
    gstar["arm_id"] = "G*"

    # F1: family only + a59 base (no b8/c1)
    fam_cands = [n for n in cands if n.startswith("a59_") or n.startswith("fam_")]
    print("=== F1 family-only ===", flush=True)
    f1 = greedy_select(bank, fam_cands)
    f1["arm_id"] = "F1"

    # V1: view only a59+b8+c1
    view_cands = [n for n in cands if n.startswith("a59_") or n.startswith("b8_") or n.startswith("c1_")]
    print("=== V1 view-only ===", flush=True)
    v1 = greedy_select(bank, view_cands)
    v1["arm_id"] = "V1"

    print("=== U0 two-stream ===", flush=True)
    u0 = nested_multistream_u0(bank)

    def gate(arm: dict) -> dict:
        delta = float(arm["val_acc_mean"] - a0["val_acc_mean"])
        pw = wilcoxon_p(arm["fold_accs"], a0["fold_accs"])
        pt = paired_t_p(arm["fold_accs"], a0["fold_accs"])
        return {
            "val_acc_mean": float(arm["val_acc_mean"]),
            "val_acc_std": float(arm.get("val_acc_std", 0)),
            "delta_vs_a0": delta,
            "wilcoxon_p": pw,
            "paired_t_p": pt,
            "pass_replace": bool(delta >= DELTA_LINE - 1e-12 and pw is not None and pw < 0.05),
            "fold_accs": arm["fold_accs"],
            "pool": arm.get("pool"),
        }

    arms = {"A0": a0, "G*": gstar, "F1": f1, "V1": v1, "U0": u0}
    gates = {k: gate(v) if k != "A0" else {"val_acc_mean": a0_mean, "role": "anchor", "fold_accs": a0_accs} for k, v in arms.items()}
    # A0 gate stub
    g_pass = bool(gates["G*"]["pass_replace"])
    # tie-break vs U0
    prefer = "G*"
    if g_pass and abs(gates["G*"]["val_acc_mean"] - gates["U0"]["val_acc_mean"]) < DELTA_LINE:
        prefer = "U0" if len(u0.get("pool") or []) <= len(gstar["pool"]) else "G*"
        # U0 has no pool list - treat as simpler
        prefer = "U0"

    decision = "REPLACE_with_G*" if g_pass else "KEEP_S0"
    if g_pass and prefer == "U0" and gates["U0"]["pass_replace"]:
        decision = "REPLACE_with_U0_simpler"
    elif g_pass:
        decision = "REPLACE_with_G*"

    doc = {
        "experiment": 38,
        "scheme_version": "v0.1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": args.run_tag,
        "candidates": list(cands.keys()),
        "candidate_paths": {k: str(v) for k, v in cands.items()},
        "arms": {
            k: {kk: vv for kk, vv in v.items() if kk != "folds"}
            for k, v in arms.items()
        },
        "gates": gates,
        "gstar_pass": g_pass,
        "decision": decision,
        "note": "A0=nested A59-4; G*=nested greedy unified pool; dgcnn→deep4 substitution",
    }
    out = exp38_out() / "replay"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"greedy_{stamp}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "greedy_latest.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp38_out() / "metrics_full_latest.json").write_text(
        json.dumps(
            {
                "experiment": 38,
                "generated_at": doc["generated_at"],
                "decision": decision,
                "gstar_pass": g_pass,
                "gates": {
                    k: {kk: gates[k].get(kk) for kk in ("val_acc_mean", "delta_vs_a0", "wilcoxon_p", "pass_replace", "pool")}
                    for k in gates
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    print("decision", decision)
    _update_registry(doc)
    return 0


def _update_registry(doc: dict) -> None:
    gates = doc["gates"]
    arms = doc["arms"]

    def row(aid: str) -> str:
        g = gates.get(aid) or {}
        if aid == "A0":
            return f"| A0 | {g.get('val_acc_mean', float('nan')):.3f} | — | — | nested A59×4 | — | ✅ |"
        pool = g.get("pool") or arms.get(aid, {}).get("pool") or "—"
        if isinstance(pool, list):
            pool = ",".join(pool)
        p = g.get("wilcoxon_p")
        ps = f"{p:.4f}" if isinstance(p, float) and p == p else "—"
        d = g.get("delta_vs_a0", 0) * 100
        return (
            f"| {aid} | {g.get('val_acc_mean', float('nan')):.3f}±{g.get('val_acc_std', 0):.3f} | "
            f"{d:+.1f}pp | {ps} | {pool} | {'✅' if g.get('pass_replace') else '否'} | ✅ |"
        )

    path_lines = []
    for s in (arms.get("G*") or {}).get("path") or []:
        gain = s.get("gain_pp")
        gain_s = "—" if gain is None else f"{gain:+.2f}"
        path_lines.append(
            f"| {s.get('step')} | {s.get('add')} | {s.get('nested_acc', float('nan')):.3f} | {gain_s} |"
        )
    path = "\n".join(path_lines)
    decision = doc.get("decision")
    text = f"""# 实验 38 · 结果登记

> 方案：[../方案.md](../方案.md) · **v0.1**  
> 状态：**已跑 D1/D2 · decision=`{decision}`** · `{doc.get('generated_at')}`  
> 对照锚 A0：**nested-S0**（Exp37 未换卷）  
> 产物：`code/train_lab/out/{OUT_ROOT_TAG}/`

## 表 0 · 门控

| 项 | 值 |
|----|-----|
| Exp37 | 已结案 · N7 未过线 → A0=nested-S0 |
| 过线 | G* 嵌套 Δ≥+1pp vs A0 且 Wilcoxon p&lt;0.05 |
| 家族替换 | dgcnn_raw(8ch) → **deep4** |

## 表 1 · 主臂

| 臂 | 嵌套 Val | vs A0 | Wilcoxon p | 池组成 | 过线 | 状态 |
|----|----------|-------|------------|--------|------|------|
{row("A0")}
{row("G*")}
{row("U0")}
{row("F1")}
{row("V1")}

## 表 2 · G* 贪心路径

| 步 | 加入 | 嵌套 Acc | 单步 Δpp |
|----|------|----------|----------|
{path}

## 决策

`{decision}` · gstar_pass={doc.get("gstar_pass")}
"""
    reg = scheme_doc() / "总结" / "结果登记表.md"
    reg.write_text(text, encoding="utf-8")
    print("updated", reg)


if __name__ == "__main__":
    raise SystemExit(main())
