# -*- coding: utf-8 -*-
"""Exp37：嵌套融合重放（主读 Wilcoxon；辅证 McNemar + cluster bootstrap）。

用法：
  python replay_nested.py
  python replay_nested.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent
from exp37_config import (  # noqa: E402
    DELTA_LINE,
    MEMBER_KEYS,
    N_BOOT,
    PREFER_TAG,
    W_B8_MAX,
    W_C1_MAX,
    a59_step,
    exp36_out,
    exp36_step,
    exp37_out,
    rankflip_step,
)

_RF = rankflip_step()
_A59 = a59_step()
_E36 = exp36_step()
for p in (_RF, _A59, _E36, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e1f_constrained import fit_e1f_constrained, fuse_with_config  # noqa: E402
from e1f_core import E1fConfig, accuracy, apply_temperature, fit_temperature  # noqa: E402
from exp35_config import FusionConstraints  # noqa: E402
from member_paths import find_all_a59_members, find_all_b8_members, n_folds_available  # noqa: E402
from metrics_secondary import (  # noqa: E402
    cluster_bootstrap_delta,
    mcnemar_exact,
    paired_t_p,
    wilcoxon_p,
)


def _load_fold(three: Path, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fd = three / f"fold{fold}"
    p = np.load(fd / "val_prob.npy")
    y = np.load(fd / "val_y.npy")
    idx = np.load(fd / "val_idx.npy") if (fd / "val_idx.npy").is_file() else np.arange(len(y))
    return p.astype(np.float32), y.astype(np.int64), idx.astype(np.int64)


def _find_c1_three() -> Path:
    root = exp36_out() / "C1"
    cands = sorted(
        root.glob("ft_conformer_*/challenge_mi_3s_45ch/run_*/three"),
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError("找不到 Exp36 C1 FT three 目录")
    return cands[-1]


def _find_exp36_b4(folder: str) -> Path | None:
    root = exp36_out() / "B8ft" / "B4" / folder
    if not root.is_dir():
        return None
    runs = sorted(root.glob("run_*/three"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _mean_probs(threes: list[Path], fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ps, y0, i0 = [], None, None
    for th in threes:
        p, y, idx = _load_fold(th, fold)
        if y0 is None:
            y0, i0 = y, idx
        elif not np.array_equal(y, y0) or not np.array_equal(idx, i0):
            raise RuntimeError(f"seed align fail {th}")
        ps.append(p)
    assert y0 is not None and i0 is not None
    return np.mean(np.stack(ps, axis=0), axis=0).astype(np.float32), y0, i0


def load_track_bank(
    member_dirs: dict[str, Path],
    n_folds: int,
    *,
    average_extra: dict[str, list[Path]] | None = None,
) -> dict:
    """每折：names, probs[list], y, idx, subject 标签（从 val 目录无法直接得 subject 时用 fold 占位）。"""
    folds = []
    for fold in range(n_folds):
        names, probs = [], []
        y_ref = idx_ref = None
        for name in MEMBER_KEYS:
            if name not in member_dirs:
                continue
            threes = [member_dirs[name]]
            if average_extra and name in average_extra:
                threes = threes + average_extra[name]
            if len(threes) == 1:
                p, y, idx = _load_fold(threes[0], fold)
            else:
                p, y, idx = _mean_probs(threes, fold)
            if y_ref is None:
                y_ref, idx_ref = y, idx
            elif not np.array_equal(y, y_ref) or not np.array_equal(idx, idx_ref):
                raise RuntimeError(f"fold{fold} {name} align fail")
            names.append(name)
            probs.append(p)
        assert y_ref is not None and idx_ref is not None
        # subject id：官方 LOSO 每折一人 → challenge:S0{fold+1}
        subj = np.array([f"challenge:S{fold+1:02d}"] * len(y_ref), dtype=object)
        folds.append(
            {
                "fold": fold,
                "names": names,
                "probs": probs,
                "y": y_ref,
                "idx": idx_ref,
                "subjects": subj,
            }
        )
    return {"n_folds": n_folds, "folds": folds}


def load_c1_bank(c1_three: Path, n_folds: int, ref_folds: list[dict]) -> dict:
    folds = []
    for fold in range(n_folds):
        p, y, idx = _load_fold(c1_three, fold)
        y_ref, idx_ref = ref_folds[fold]["y"], ref_folds[fold]["idx"]
        if not np.array_equal(y, y_ref) or not np.array_equal(idx, idx_ref):
            raise RuntimeError(f"C1 fold{fold} align fail vs A59")
        folds.append(
            {
                "fold": fold,
                "names": ["c1_conformer"],
                "probs": [p],
                "y": y,
                "idx": idx,
                "subjects": ref_folds[fold]["subjects"],
            }
        )
    return {"n_folds": n_folds, "folds": folds}


def _concat_member_probs(folds: list[dict]) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    names = folds[0]["names"]
    for fr in folds[1:]:
        if fr["names"] != names:
            raise RuntimeError("member name order mismatch across folds")
    probs = [
        np.concatenate([fr["probs"][i] for fr in folds], axis=0) for i in range(len(names))
    ]
    y = np.concatenate([fr["y"] for fr in folds], axis=0)
    return names, probs, y


def _fit_e1f(names: list[str], probs: list[np.ndarray], y: np.ndarray) -> E1fConfig:
    return fit_e1f_constrained(names, probs, y, FusionConstraints(name="nested_unc"))


def _apply_e1f(fold: dict, cfg: E1fConfig) -> np.ndarray:
    # cfg.member_names 顺序
    name_to_p = {n: p for n, p in zip(fold["names"], fold["probs"])}
    ordered = [name_to_p[n] for n in cfg.member_names]
    return fuse_with_config(ordered, cfg)


def _fit_streams(
    stream_probs: list[np.ndarray],
    names: list[str],
    y: np.ndarray,
    *,
    w_caps: dict[str, float],
) -> tuple[E1fConfig, np.ndarray]:
    temps = [fit_temperature(p, y) for p in stream_probs]
    cals = [apply_temperature(p, t) for p, t in zip(stream_probs, temps)]
    k = len(stream_probs)
    if k == 1:
        cfg = E1fConfig(
            member_names=names,
            temperatures=temps,
            weights=[1.0],
            smooth_radius=0,
            val_acc=float(accuracy(cals[0], y)),
        )
        return cfg, cals[0]
    if k == 2:
        cap_b = float(w_caps.get(names[1], 1.0))
        best_w, best_acc = 0.0, -1.0
        for w_b in np.arange(0.0, cap_b + 1e-9, 0.05):
            fused = (1.0 - float(w_b)) * cals[0] + float(w_b) * cals[1]
            acc = accuracy(fused, y)
            if acc > best_acc + 1e-12:
                best_acc, best_w = acc, float(w_b)
        weights = [1.0 - best_w, best_w]
        fused = weights[0] * cals[0] + weights[1] * cals[1]
        return (
            E1fConfig(
                member_names=names,
                temperatures=temps,
                weights=weights,
                smooth_radius=0,
                val_acc=float(best_acc),
            ),
            fused,
        )
    # k==3
    cap1 = float(w_caps.get(names[1], 1.0))
    cap2 = float(w_caps.get(names[2], 1.0))
    best, best_acc = (1.0, 0.0, 0.0), -1.0
    for w1 in np.arange(0.0, cap1 + 1e-9, 0.05):
        for w2 in np.arange(0.0, cap2 + 1e-9, 0.05):
            if w1 + w2 > 1.0 + 1e-9:
                continue
            w0 = 1.0 - w1 - w2
            fused = w0 * cals[0] + w1 * cals[1] + w2 * cals[2]
            acc = accuracy(fused, y)
            if acc > best_acc + 1e-12:
                best_acc = acc
                best = (float(w0), float(w1), float(w2))
    fused = best[0] * cals[0] + best[1] * cals[1] + best[2] * cals[2]
    return (
        E1fConfig(
            member_names=names,
            temperatures=temps,
            weights=list(best),
            smooth_radius=0,
            val_acc=float(best_acc),
        ),
        fused,
    )


def _apply_streams(
    stream_probs: list[np.ndarray],
    cfg: E1fConfig,
) -> np.ndarray:
    cals = [apply_temperature(p, t) for p, t in zip(stream_probs, cfg.temperatures)]
    w = np.asarray(cfg.weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    return np.tensordot(w, np.stack(cals, axis=0), axes=(0, 0)).astype(np.float32)


def nested_e1f_track(bank: dict) -> dict:
    """N0：仅 A59 四成员嵌套 E1f。"""
    n_folds = bank["n_folds"]
    oof_probs, oof_y, oof_subj, fold_accs, fold_meta = [], [], [], [], []
    for k in range(n_folds):
        fit_folds = [bank["folds"][f] for f in range(n_folds) if f != k]
        names, probs, y_fit = _concat_member_probs(fit_folds)
        cfg = _fit_e1f(names, probs, y_fit)
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
                "w_extra": 0.0,
                "stream_weights": {"a59_e1f": 1.0},
            }
        )
        print(f"  N0 fold{k} acc={acc:.4f}", flush=True)
    return _pack_arm("N0", "nested-S0 A59 E1f", fold_accs, fold_meta, oof_probs, oof_y, oof_subj)


def nested_multistream(
    *,
    arm_id: str,
    desc: str,
    banks: list[dict],
    stream_names: list[str],
    w_caps: dict[str, float],
) -> dict:
    """
    banks: 与 stream 对齐的成员库（单成员库如 C1 也行）。
    嵌套：每流在 fit 折上拟 E1f → 在 fit 折上流概率上拟跨流 w → 评估折应用。
    """
    n_folds = banks[0]["n_folds"]
    oof_probs, oof_y, oof_subj, fold_accs, fold_meta = [], [], [], [], []
    for k in range(n_folds):
        fit_idx = [f for f in range(n_folds) if f != k]
        stream_cfgs: list[E1fConfig] = []
        # 每流嵌套 E1f
        for bank in banks:
            fit_folds = [bank["folds"][f] for f in fit_idx]
            names, probs, y_fit = _concat_member_probs(fit_folds)
            stream_cfgs.append(_fit_e1f(names, probs, y_fit))

        # fit 折上流概率
        stream_fit = [[] for _ in banks]
        y_parts = []
        for f in fit_idx:
            y_parts.append(banks[0]["folds"][f]["y"])
            for si, bank in enumerate(banks):
                stream_fit[si].append(_apply_e1f(bank["folds"][f], stream_cfgs[si]))
        y_fit = np.concatenate(y_parts, axis=0)
        p_fit = [np.concatenate(parts, axis=0) for parts in stream_fit]

        cfg_s, _ = _fit_streams(p_fit, stream_names, y_fit, w_caps=w_caps)

        # eval 折
        p_eval = [_apply_e1f(bank["folds"][k], stream_cfgs[si]) for si, bank in enumerate(banks)]
        pred = _apply_streams(p_eval, cfg_s)
        yk = banks[0]["folds"][k]["y"]
        acc = float(accuracy(pred, yk))
        fold_accs.append(acc)
        oof_probs.append(pred)
        oof_y.append(yk)
        oof_subj.append(banks[0]["folds"][k]["subjects"])
        wmap = {n: float(w) for n, w in zip(cfg_s.member_names, cfg_s.weights)}
        w_extra = float(sum(w for n, w in wmap.items() if n != stream_names[0]))
        fold_meta.append(
            {
                "fold": k,
                "acc": acc,
                "stream_e1f": [c.to_dict() for c in stream_cfgs],
                "stream_fusion": cfg_s.to_dict(),
                "stream_weights": wmap,
                "w_extra": w_extra,
                "collapsed_to_a59": bool(w_extra < 1e-9),
            }
        )
        print(
            f"  {arm_id} fold{k} acc={acc:.4f} w={wmap} collapse={w_extra < 1e-9}",
            flush=True,
        )
    return _pack_arm(arm_id, desc, fold_accs, fold_meta, oof_probs, oof_y, oof_subj)


def _pack_arm(arm_id, desc, fold_accs, fold_meta, oof_probs, oof_y, oof_subj) -> dict:
    return {
        "arm_id": arm_id,
        "desc": desc,
        "val_acc_mean": float(np.mean(fold_accs)),
        "val_acc_std": float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0,
        "fold_accs": [float(x) for x in fold_accs],
        "folds": fold_meta,
        "_oof_probs": oof_probs,
        "_oof_y": oof_y,
        "_oof_subjects": oof_subj,
    }


def _gate_vs_n0(arm: dict, n0: dict) -> dict:
    accs = arm["fold_accs"]
    base = n0["fold_accs"]
    mean = float(arm["val_acc_mean"])
    base_mean = float(n0["val_acc_mean"])
    delta = mean - base_mean
    p_w = wilcoxon_p(accs, base)
    p_t = paired_t_p(accs, base)
    # OOF preds
    yhat_a = np.concatenate([p.argmax(1) for p in arm["_oof_probs"]])
    yhat_0 = np.concatenate([p.argmax(1) for p in n0["_oof_probs"]])
    y = np.concatenate(arm["_oof_y"])
    subj = np.concatenate(arm["_oof_subjects"])
    mcn = mcnemar_exact(yhat_a, yhat_0, y)
    # McNemar 方向：这里 A=arm, B=N0；我们要的是 arm 相对 N0
    # 上面 mcnemar(a,b)：b=A对B错 → arm对 N0错；c=arm错 N0对
    # 对「arm 更好」看 c vs b 的对称；精确 p 一样。另算 arm vs N0 正确率差的 bootstrap：
    corr_arm = (yhat_a == y).astype(np.float64)
    corr_n0 = (yhat_0 == y).astype(np.float64)
    boot = cluster_bootstrap_delta(corr_arm, corr_n0, subj, n_boot=N_BOOT)
    fold0 = arm["folds"][0] if arm["folds"] else {}
    return {
        "val_acc_mean": mean,
        "val_acc_std": float(arm["val_acc_std"]),
        "nested_delta_vs_n0": delta,
        "wilcoxon_p": p_w,
        "paired_t_p": p_t,
        "pass_delta": bool(delta >= DELTA_LINE - 1e-12),
        "pass_wilcoxon": bool(p_w is not None and p_w < 0.05),
        "pass_replace": bool(delta >= DELTA_LINE - 1e-12 and p_w is not None and p_w < 0.05),
        "fold0_delta": float(accs[0] - base[0]),
        "fold0_w_extra": float(fold0.get("w_extra", 0.0)),
        "fold0_collapsed": bool(fold0.get("collapsed_to_a59", False)),
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
        "fold_deltas_pp": [float((a - b) * 100) for a, b in zip(accs, base)],
    }


def save_oof(arm: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aid = arm["arm_id"]
    np.save(out_dir / f"oof_{aid}_prob.npy", np.concatenate(arm["_oof_probs"], axis=0))
    np.save(out_dir / f"oof_{aid}_y.npy", np.concatenate(arm["_oof_y"], axis=0))
    np.save(out_dir / f"oof_{aid}_subjects.npy", np.concatenate(arm["_oof_subjects"], axis=0))


def main() -> int:
    global N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer-tag", default=PREFER_TAG)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--skip-n7b", action="store_true")
    args = ap.parse_args()
    N_BOOT = int(args.n_boot)

    a59 = find_all_a59_members(prefer_tag=args.prefer_tag)
    b8 = find_all_b8_members(arm="ft", prefer_tag=args.prefer_tag)
    if len(a59) < 4 or len(b8) < 4:
        raise SystemExit(f"成员不足 a59={list(a59)} b8={list(b8)}")
    n_folds = min(n_folds_available(a59), n_folds_available(b8), 6)
    c1_three = _find_c1_three()
    print(f"n_folds={n_folds} c1={c1_three}", flush=True)

    bank_a = load_track_bank(a59, n_folds)
    bank_b = load_track_bank(b8, n_folds)
    bank_c = load_c1_bank(c1_three, n_folds, bank_a["folds"])

    # N7b extras
    b8_extra: dict[str, list[Path]] = {}
    for name in MEMBER_KEYS:
        extras = []
        for folder in (f"{name}_s43", f"{name}_s44"):
            th = _find_exp36_b4(folder)
            if th is not None:
                extras.append(th)
        if extras:
            b8_extra[name] = extras
    bank_b_avg = None
    if b8_extra and not args.skip_n7b:
        bank_b_avg = load_track_bank(b8, n_folds, average_extra=b8_extra)
        print(f"N7b extras: { {k: len(v) for k, v in b8_extra.items()} }", flush=True)

    arms: dict[str, dict] = {}
    print("=== N0 ===", flush=True)
    arms["N0"] = nested_e1f_track(bank_a)

    print("=== N7 ===", flush=True)
    arms["N7"] = nested_multistream(
        arm_id="N7",
        desc="nested M7 A59×B8 w_B8≤0.4",
        banks=[bank_a, bank_b],
        stream_names=["a59_e1f", "b8_ft_e1f"],
        w_caps={"b8_ft_e1f": W_B8_MAX},
    )

    if bank_b_avg is not None:
        print("=== N7b ===", flush=True)
        arms["N7b"] = nested_multistream(
            arm_id="N7b",
            desc="nested M7b B8 multiseed-avg",
            banks=[bank_a, bank_b_avg],
            stream_names=["a59_e1f", "b8_ft_e1f"],
            w_caps={"b8_ft_e1f": W_B8_MAX},
        )

    print("=== N7_AC ===", flush=True)
    arms["N7_AC"] = nested_multistream(
        arm_id="N7_AC",
        desc="nested A59×C1 w_C1≤0.5",
        banks=[bank_a, bank_c],
        stream_names=["a59_e1f", "c1_conformer"],
        w_caps={"c1_conformer": W_C1_MAX},
    )

    print("=== N7_ABC ===", flush=True)
    arms["N7_ABC"] = nested_multistream(
        arm_id="N7_ABC",
        desc="nested A59×B8×C1",
        banks=[bank_a, bank_b, bank_c],
        stream_names=["a59_e1f", "b8_ft_e1f", "c1_conformer"],
        w_caps={"b8_ft_e1f": W_B8_MAX, "c1_conformer": W_C1_MAX},
    )

    n0 = arms["N0"]
    gates = {}
    for aid, arm in arms.items():
        if aid == "N0":
            gates[aid] = {
                "val_acc_mean": arm["val_acc_mean"],
                "val_acc_std": arm["val_acc_std"],
                "fold_accs": arm["fold_accs"],
                "role": "anchor",
            }
        else:
            gates[aid] = _gate_vs_n0(arm, n0)
            print(
                f"GATE {aid}: nested={gates[aid]['val_acc_mean']:.4f} "
                f"Δ={gates[aid]['nested_delta_vs_n0']*100:+.2f}pp "
                f"W={gates[aid]['wilcoxon_p']} pass={gates[aid]['pass_replace']}",
                flush=True,
            )

    n7_pass = bool(gates.get("N7", {}).get("pass_replace"))
    upgrade = False
    if n7_pass and "N7_ABC" in gates:
        upgrade = (
            float(gates["N7_ABC"]["val_acc_mean"]) - float(gates["N7"]["val_acc_mean"])
            >= DELTA_LINE - 1e-12
        )

    if n7_pass:
        decision = "REPLACE_with_N7"
        status = "PASS_N7"
    elif float(gates.get("N7", {}).get("nested_delta_vs_n0", 0)) >= DELTA_LINE and not n7_pass:
        decision = "KEEP_S0_weak_positive"
        status = "DELTA_OK_WILCOXON_FAIL"
    else:
        decision = "KEEP_S0_shrink_or_fail"
        status = "NO_REPLACE"

    out = exp37_out()
    replay = out / "replay"
    preds = out / "preds"
    replay.mkdir(parents=True, exist_ok=True)
    for arm in arms.values():
        save_oof(arm, preds)

    # strip private arrays for json
    arms_pub = {}
    for aid, arm in arms.items():
        arms_pub[aid] = {
            k: v
            for k, v in arm.items()
            if not k.startswith("_")
        }

    doc = {
        "experiment": 37,
        "scheme_version": "v0.2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prefer_tag": args.prefer_tag,
        "n_folds": n_folds,
        "c1_run": str(c1_three),
        "delta_line": DELTA_LINE,
        "n_boot": N_BOOT,
        "arms": arms_pub,
        "gates": gates,
        "n7_pass_replace": n7_pass,
        "upgrade_discuss_n7_abc": upgrade,
        "decision": decision,
        "status": status,
        "note": "Primary=nested Wilcoxon N7 vs N0; secondary=McNemar+subject cluster bootstrap",
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (replay / f"nested_{stamp}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (replay / "nested_latest.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "metrics_full_latest.json").write_text(
        json.dumps(
            {
                "experiment": 37,
                "generated_at": doc["generated_at"],
                "status": status,
                "decision": decision,
                "n7_pass_replace": n7_pass,
                "gates": {
                    k: {
                        kk: gates[k].get(kk)
                        for kk in (
                            "val_acc_mean",
                            "nested_delta_vs_n0",
                            "wilcoxon_p",
                            "paired_t_p",
                            "pass_replace",
                            "fold0_collapsed",
                            "mcnemar",
                            "cluster_bootstrap",
                        )
                        if kk in gates[k] or k != "N0"
                    }
                    for k in gates
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("decision", decision, "status", status)
    print("wrote", replay / "nested_latest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
