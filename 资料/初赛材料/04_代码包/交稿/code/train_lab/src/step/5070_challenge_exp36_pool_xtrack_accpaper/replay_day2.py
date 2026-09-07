# -*- coding: utf-8 -*-
"""Day2：C1 单模 vs S0；A59×C1；A59×B8×C1 三流融合。

用法：
  python replay_day2.py
  python replay_day2.py --c1-run <path/to/three>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

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
from e1f_core import E1fConfig, accuracy, apply_temperature, fit_temperature  # noqa: E402
from exp35_config import FusionConstraints, MEMBER_KEYS  # noqa: E402
from member_paths import find_all_a59_members, find_all_b8_members, n_folds_available  # noqa: E402
from metrics_three import three_class_report  # noqa: E402

W_C1_MAX = 0.50


def _wilcoxon_p(a: list[float], b: list[float]) -> float | None:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return None


def _gate(mean: float, fold_accs: list[float], s0_folds: list[float]) -> dict:
    delta = float(mean - ANCHOR_S0)
    p = _wilcoxon_p(fold_accs, s0_folds) if s0_folds and len(fold_accs) == len(s0_folds) else None
    return {
        "val_acc_mean": mean,
        "delta_vs_s0": delta,
        "wilcoxon_p": p,
        "pass_replace": bool(mean >= REPLACE_LINE and p is not None and p < 0.05),
    }


def _load_pool(member_dirs: dict[str, Path], fold: int, pool: list[str]):
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
            if not np.array_equal(y, y_ref) or not np.array_equal(idx, idx_ref):
                raise RuntimeError(f"fold{fold} {name} align mismatch")
        names.append(name)
        probs.append(p)
    assert y_ref is not None and idx_ref is not None
    return names, probs, y_ref, idx_ref


def _fuse_track(member_dirs: dict[str, Path], fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names, probs, y, idx = _load_pool(member_dirs, fold, list(MEMBER_KEYS))
    cfg = fit_e1f_constrained(names, probs, y, FusionConstraints(name="unconstrained"))
    return fuse_with_config(probs, cfg), y, idx


def _fit_streams(
    probs: list[np.ndarray],
    names: list[str],
    y: np.ndarray,
    *,
    w_caps: dict[str, float] | None = None,
) -> tuple[E1fConfig, np.ndarray]:
    """网格搜权重（步长 0.05），带可选上界；温度各自拟合。"""
    w_caps = w_caps or {}
    temps = [fit_temperature(p, y) for p in probs]
    cals = [apply_temperature(p, t) for p, t in zip(probs, temps)]
    k = len(probs)
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

    # k==3：网格 w1,w2；w0=1-w1-w2；各自 cap
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

def find_latest_c1_three() -> Path | None:
    root = exp36_out() / "C1"
    if not root.is_dir():
        return None
    cands = sorted(root.glob("ft_conformer_*/challenge_mi_3s_45ch/run_*/three"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def load_c1_fold(c1_three: Path, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fd = c1_three / f"fold{fold}"
    return (
        np.load(fd / "val_prob.npy"),
        np.load(fd / "val_y.npy"),
        np.load(fd / "val_idx.npy") if (fd / "val_idx.npy").is_file() else np.arange(
            len(np.load(fd / "val_y.npy"))
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-run", type=str, default="")
    ap.add_argument("--prefer-tag", default=PREFER_TAG)
    args = ap.parse_args()

    c1_three = Path(args.c1_run) if args.c1_run else find_latest_c1_three()
    if c1_three is None or not c1_three.is_dir():
        raise SystemExit("找不到 C1 FT run（--c1-run）")

    a59 = find_all_a59_members(prefer_tag=args.prefer_tag)
    b8 = find_all_b8_members(arm="ft", prefer_tag=args.prefer_tag)
    n_folds = min(n_folds_available(a59), n_folds_available(b8), 6)
    # C1 folds
    c1_folds = sum(1 for i in range(6) if (c1_three / f"fold{i}" / "val_prob.npy").is_file())
    n_folds = min(n_folds, c1_folds)
    print(f"n_folds={n_folds} c1={c1_three}", flush=True)

    s0_accs, c1_accs = [], []
    m_ac_accs, m_abc_accs = [], []
    fold_details = []

    for fold in range(n_folds):
        p_a, y_a, idx_a = _fuse_track(a59, fold)
        p_b, y_b, idx_b = _fuse_track(b8, fold)
        p_c, y_c, idx_c = load_c1_fold(c1_three, fold)
        if not (np.array_equal(y_a, y_b) and np.array_equal(idx_a, idx_b)):
            raise RuntimeError(f"fold{fold} A59/B8 align fail")
        if not (np.array_equal(y_a, y_c) and np.array_equal(idx_a, idx_c)):
            raise RuntimeError(
                f"fold{fold} C1 align fail vs A59 "
                f"len_a={len(y_a)} len_c={len(y_c)}"
            )

        s0_acc = float(accuracy(p_a, y_a))
        c1_acc = float(accuracy(p_c, y_c))
        _, fused_ac = _fit_streams(
            [p_a, p_c],
            ["a59_e1f", "c1_conformer"],
            y_a,
            w_caps={"c1_conformer": W_C1_MAX},
        )
        _, fused_abc = _fit_streams(
            [p_a, p_b, p_c],
            ["a59_e1f", "b8_ft_e1f", "c1_conformer"],
            y_a,
            w_caps={"b8_ft_e1f": W_B8_MAX, "c1_conformer": W_C1_MAX},
        )
        ac_acc = float(accuracy(fused_ac, y_a))
        abc_acc = float(accuracy(fused_abc, y_a))
        s0_accs.append(s0_acc)
        c1_accs.append(c1_acc)
        m_ac_accs.append(ac_acc)
        m_abc_accs.append(abc_acc)
        fold_details.append(
            {
                "fold": fold,
                "S0": s0_acc,
                "C1": c1_acc,
                "M7_AC": ac_acc,
                "M7_ABC": abc_acc,
                "n": int(len(y_a)),
            }
        )
        print(
            f"fold{fold} S0={s0_acc:.4f} C1={c1_acc:.4f} "
            f"A×C={ac_acc:.4f} A×B×C={abc_acc:.4f}",
            flush=True,
        )

    def pack(name: str, accs: list[float]) -> dict:
        mean = float(np.mean(accs))
        std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        g = _gate(mean, accs, s0_accs)
        return {
            "name": name,
            "val_acc_mean": mean,
            "val_acc_std": std,
            "fold_accs": accs,
            **g,
        }

    arms = {
        "S0": pack("S0", s0_accs),
        "C1": pack("C1", c1_accs),
        "M7_AC": pack("M7_AC", m_ac_accs),
        "M7_ABC": pack("M7_ABC", m_abc_accs),
    }
    gates = {k: {kk: arms[k][kk] for kk in ("val_acc_mean", "delta_vs_s0", "wilcoxon_p", "pass_replace")} for k in arms}
    any_pass = any(gates[k]["pass_replace"] for k in ("C1", "M7_AC", "M7_ABC"))
    doc = {
        "experiment": 36,
        "stage": "day2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "c1_run": str(c1_three),
        "n_folds": n_folds,
        "replace_line": REPLACE_LINE,
        "arms": arms,
        "gates": gates,
        "any_pass_replace": any_pass,
        "folds": fold_details,
        "note": "C1=OpenBMI45→official45 conformer FT; M7_AC=A59×C1; M7_ABC=A59×B8×C1",
    }
    out = exp36_out() / "replay"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"day2_{stamp}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "day2_latest.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: gates[k] for k in gates}, ensure_ascii=False, indent=2))
    print("any_pass_replace", any_pass)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
