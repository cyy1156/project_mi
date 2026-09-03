# -*- coding: utf-8 -*-
"""轨 F / M / D 回放网格。

用法：
  python replay_fusion_grid.py --suite FM          # A59 · F0–F7 + M*
  python replay_fusion_grid.py --suite D           # B8 · D0–D4
  python replay_fusion_grid.py --arms F0,M1,M2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent
_A59 = _STEP.parent / "5070_challenge_mi_59ch_accpaper"
# A59 在后、本包在前，保证 arm_defs / e1f_constrained / member_paths 命中 Exp35
if str(_A59) not in sys.path:
    sys.path.insert(0, str(_A59))
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

from arm_defs import D_ARMS, F_ARMS, M_ARMS, resolve_cons  # noqa: E402
from e1f_constrained import fit_e1f_constrained, fuse_with_config  # noqa: E402
from exp35_config import ANCHOR_CONFORMER_A59, ANCHOR_E1F_A59, exp35_out  # noqa: E402
from member_paths import (  # noqa: E402
    find_all_a59_members,
    find_all_b8_members,
    n_folds_available,
)
from metrics_three import three_class_report  # noqa: E402


def _load_fold_probs(
    member_dirs: dict[str, Path],
    pool: list[str],
    fold: int,
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    names: list[str] = []
    probs: list[np.ndarray] = []
    y_ref: np.ndarray | None = None
    for name in pool:
        if name not in member_dirs:
            raise KeyError(f"缺成员 {name}")
        fd = member_dirs[name] / f"fold{fold}"
        p = np.load(fd / "val_prob.npy")
        y = np.load(fd / "val_y.npy")
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise RuntimeError(f"fold{fold} {name} val_y 不一致")
        names.append(name)
        probs.append(p)
    assert y_ref is not None
    return names, probs, y_ref


def _summarize_weights(folds: list[dict], name: str = "eegnet") -> dict:
    vals = []
    for fr in folds:
        cfg = fr["config"]
        names = cfg["member_names"]
        ws = cfg["weights"]
        if name in names:
            vals.append(float(ws[names.index(name)]))
        else:
            vals.append(0.0)
    if not vals:
        return {"mean": None, "max": None}
    return {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "per_fold": vals}


def run_arm(
    arm_id: str,
    arm_def: dict,
    member_dirs: dict[str, Path],
    n_folds: int,
) -> dict:
    pool = list(arm_def["pool"])
    fold_recs = []
    for fold in range(n_folds):
        names, probs, y = _load_fold_probs(member_dirs, pool, fold)
        cons = resolve_cons(arm_def, names)
        cfg = fit_e1f_constrained(names, probs, y, cons)
        fused = fuse_with_config(probs, cfg)
        report = three_class_report(y, fused.argmax(axis=1))
        fold_recs.append(
            {
                "fold": fold,
                "config": cfg.to_dict(),
                "val_metrics": report,
                "member_accs_calibrated": getattr(cfg, "_member_accs", None),
                "relaxed": bool(getattr(cfg, "_rankflip_relaxed", False)),
            }
        )
    accs = [float(r["val_metrics"]["acc"]) for r in fold_recs]
    return {
        "arm_id": arm_id,
        "desc": arm_def.get("desc", ""),
        "pool": pool,
        "n_folds": n_folds,
        "val_acc_mean": float(np.mean(accs)),
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "w_eegnet": _summarize_weights(fold_recs, "eegnet"),
        "w_conformer": _summarize_weights(fold_recs, "conformer"),
        "folds": fold_recs,
    }


def decide_f(results: dict[str, dict], f0_acc: float) -> dict:
    """方案 §7.5。"""
    promoted = []
    robust = []
    for aid, rec in results.items():
        if not aid.startswith("F") or aid == "F0":
            continue
        if aid in ("F1", "F2"):
            continue  # 附报
        mean = float(rec["val_acc_mean"])
        max_eeg = rec["w_eegnet"]["max"]
        if mean >= f0_acc + 0.010:
            promoted.append(aid)
        elif mean >= f0_acc - 0.005 and max_eeg is not None and max_eeg < 0.35:
            robust.append(aid)
    best = None
    pool = promoted or robust
    if pool:
        best = max(pool, key=lambda a: results[a]["val_acc_mean"])
    return {
        "f0_acc": f0_acc,
        "sanity_ok": abs(f0_acc - ANCHOR_E1F_A59) <= 0.005,
        "promoted_plus1pp": promoted,
        "robust": robust,
        "s2_candidate": best,
    }


def decide_m(results: dict[str, dict], f0_acc: float) -> dict:
    m1 = results.get("M1", {}).get("val_acc_mean")
    m2 = results.get("M2", {}).get("val_acc_mean")
    m2c = results.get("M2c", {}).get("val_acc_mean")
    m5 = results.get("M5", {}).get("val_acc_mean")
    notes = []
    if m1 is not None:
        notes.append(
            f"M1 sanity vs anchor: {m1:.4f} vs {ANCHOR_CONFORMER_A59:.4f} "
            f"({'OK' if abs(m1 - ANCHOR_CONFORMER_A59) <= 0.005 else 'CHECK'})"
        )
        if m1 >= f0_acc - 0.005:
            notes.append("M1 ≥ F0−0.005 → conformer 进交卷决赛")
    s3 = None
    for cand, val in (("M2c", m2c), ("M2", m2)):
        if val is None:
            continue
        floor = max(f0_acc, m1 or -1.0) - 0.005
        if val >= f0_acc + 0.010:
            s3 = cand
            notes.append(f"{cand} ≥ F0+0.010 → 强采纳缩池")
            break
        if val >= floor and s3 is None:
            s3 = cand
            notes.append(f"{cand} 不掉点 → 缩池候选")
    if m5 is not None and m1 is not None and m5 >= m1:
        notes.append("WARNING: M5≥M1 → 排名翻转假说受冲击，需审计")
    return {
        "s3_candidate": s3,
        "m1_enter_final": bool(m1 is not None and m1 >= f0_acc - 0.005),
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("FM", "D", "F", "M", "all"), default="FM")
    ap.add_argument("--arms", default="", help="逗号分隔臂 ID，覆盖 suite")
    ap.add_argument("--prefer-tag", default="full_20260902_1930")
    ap.add_argument("--max-folds", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected: list[tuple[str, dict, str]] = []  # id, def, track_key

    def add_from(mapping: dict, track: str, ids: list[str] | None = None):
        for k, v in mapping.items():
            if ids is not None and k not in ids:
                continue
            selected.append((k, v, track))

    if args.arms.strip():
        want = {x.strip() for x in args.arms.split(",") if x.strip()}
        add_from(F_ARMS, "a59", list(want & set(F_ARMS)))
        add_from(M_ARMS, "a59", list(want & set(M_ARMS)))
        add_from(D_ARMS, "b8", list(want & set(D_ARMS)))
    else:
        if args.suite in ("FM", "F", "all"):
            add_from(F_ARMS, "a59")
        if args.suite in ("FM", "M", "all"):
            add_from(M_ARMS, "a59")
        if args.suite in ("D", "all"):
            add_from(D_ARMS, "b8")

    # 缓存 member_dirs
    cache: dict[str, dict[str, Path]] = {}
    results: dict[str, dict] = {}

    for arm_id, arm_def, track in selected:
        if track == "a59":
            key = "a59"
            if key not in cache:
                cache[key] = find_all_a59_members(prefer_tag=args.prefer_tag)
            member_dirs = cache[key]
        else:
            b8_arm = arm_def["b8_arm"]
            key = f"b8_{b8_arm}"
            if key not in cache:
                cache[key] = find_all_b8_members(arm=b8_arm, prefer_tag=args.prefer_tag)
            member_dirs = cache[key]

        missing = [m for m in arm_def["pool"] if m not in member_dirs]
        if missing:
            print(f"[skip] {arm_id} 缺成员 {missing}")
            continue
        n_folds = min(args.max_folds, n_folds_available(member_dirs))
        print(f"=== {arm_id} pool={arm_def['pool']} folds={n_folds} ===")
        rec = run_arm(arm_id, arm_def, member_dirs, n_folds)
        rec["track"] = track
        if track != "a59":
            rec["b8_arm"] = arm_def.get("b8_arm")
        results[arm_id] = rec
        print(
            f"  Val {rec['val_acc_mean']:.4f}±{rec['val_acc_std']:.4f} "
            f"w_eeg_max={rec['w_eegnet']['max']} w_c_max={rec['w_conformer']['max']}"
        )

    f0 = results.get("F0") or results.get("M0")
    f0_acc = float(f0["val_acc_mean"]) if f0 else ANCHOR_E1F_A59
    decision = {
        "F": decide_f(results, f0_acc) if any(k.startswith("F") for k in results) else None,
        "M": decide_m(results, f0_acc) if any(k.startswith("M") for k in results) else None,
    }

    payload = {
        "experiment": 35,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "suite": args.suite,
        "prefer_tag": args.prefer_tag,
        "anchor_e1f_a59": ANCHOR_E1F_A59,
        "anchor_conformer_a59": ANCHOR_CONFORMER_A59,
        "results": results,
        "decision": decision,
        "member_dirs_used": {k: {m: str(p) for m, p in v.items()} for k, v in cache.items()},
    }
    out = args.out or (exp35_out() / "replay" / f"replay_{args.suite}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # 最新指针
    latest = out.parent / f"replay_{args.suite}_latest.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", out)
    print("decision", json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
