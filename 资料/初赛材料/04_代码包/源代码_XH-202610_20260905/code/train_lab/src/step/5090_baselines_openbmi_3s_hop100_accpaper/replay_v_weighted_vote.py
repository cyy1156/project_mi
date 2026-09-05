"""方案 24 · 臂 V：置信/时间加权投票（val 选 γ,κ → test 一次）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prob_dump import merge_prob_dumps
from t0_sec import t0_norm


def _trial_group_ids(subjects: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    """(subject, trial_id) → 0..G-1。"""
    keys = np.asarray(subjects, dtype=str) + ":" + np.asarray(trial_ids, dtype=str)
    _, inv = np.unique(keys, return_inverse=True)
    return inv.astype(np.int64)


def _trial_weighted_paper_ok(
    mask: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    gamma: float,
    kappa: float,
) -> float:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0.0
    pred = data["pred"][m]
    y = data["y"][m]
    pmax = data["p_max"][m]
    tn = t0_norm(data["t0_sec"][m])
    subj = data["subject"][m]
    tid = data["trial_id"][m]

    w = np.power(pmax, gamma) * np.exp(kappa * tn)
    correct = (pred == y).astype(np.float64)
    grp = _trial_group_ids(subj, tid)
    n_g = int(grp.max()) + 1
    sum_wc = np.bincount(grp, weights=w * correct, minlength=n_g)
    sum_w = np.bincount(grp, weights=w, minlength=n_g)
    rates = sum_wc / np.maximum(sum_w, 1e-12)
    return float((rates > 0.5).mean())


def grid_search(
    data: dict[str, np.ndarray],
    *,
    split: str,
    grid: tuple[int, ...] = (0, 1, 2),
) -> tuple[float, float, float]:
    m = data["split"] == split
    best_g, best_k, best_acc = 0.0, 0.0, -1.0
    for g in grid:
        for k in grid:
            acc = _trial_weighted_paper_ok(m, data, gamma=float(g), kappa=float(k))
            if acc > best_acc:
                best_acc, best_g, best_k = acc, float(g), float(k)
    return best_g, best_k, best_acc


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme24 V · weighted vote replay")
    p.add_argument("--run-dir", type=Path, required=True, help="S3 run .../three")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    dumps = sorted(run_dir.glob("fold*/prob_dump_three.csv"))
    if not dumps:
        dumps = sorted(run_dir.glob("fold*/prob_dump_three_*.csv"))
    if not dumps:
        raise SystemExit(f"no prob dumps under {run_dir}; run dump-probs first")
    print(f"[V] loading {len(dumps)} fold dumps …", flush=True)
    data = merge_prob_dumps(dumps)
    print(f"[V] merged n={len(data['y'])} val={(data['split']=='val').sum()} test={(data['split']=='test').sum()}", flush=True)

    g, k, val_acc = grid_search(data, split="val")
    test_acc = _trial_weighted_paper_ok(
        data["split"] == "test", data, gamma=g, kappa=k
    )
    uni_val = _trial_weighted_paper_ok(
        data["split"] == "val", data, gamma=0.0, kappa=0.0
    )
    uni_test = _trial_weighted_paper_ok(
        data["split"] == "test", data, gamma=0.0, kappa=0.0
    )

    out = {
        "run_dir": str(run_dir),
        "gamma": g,
        "kappa": k,
        "val_acc_paper_weighted": val_acc,
        "test_acc_paper_weighted": test_acc,
        "val_acc_paper_uniform": uni_val,
        "test_acc_paper_uniform": uni_test,
        "delta_test_pp": (test_acc - uni_test) * 100.0,
    }
    print(json.dumps(out, indent=2))
    out_path = args.out or (run_dir / "replay_v_weighted_vote.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
