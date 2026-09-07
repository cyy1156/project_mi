# -*- coding: utf-8 -*-
"""轨 H · shallow 配方有限重训（P1）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_STEP = Path(__file__).resolve().parent
_A59 = _STEP.parent / "5070_challenge_mi_59ch_accpaper"
_PARENT = _STEP.parent
for p in (_A59, _PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

import shared_hparams as a59_hp  # noqa: E402

from baseline_shallow import build_model  # noqa: E402
from data_paths import resolve_data  # noqa: E402
from exp35_config import (  # noqa: E402
    ANCHOR_CONFORMER_A59,
    ANCHOR_SHALLOW_A59,
    exp35_out,
)
from loso import iter_loso6  # noqa: E402
from task_runner import train_one_fold  # noqa: E402

SharedTrainHP = a59_hp.SharedTrainHP
A59_SHARED = a59_hp.SHARED


def _hp_variant(arm: str, seed: int | None = None) -> SharedTrainHP:
    base = asdict(A59_SHARED)
    if arm == "H0":
        pass
    elif arm == "H1":
        base["grad_accum"] = int(base["grad_accum"]) * 2
        base["protocol"] = str(base["protocol"]) + " H1_accumx2"
    elif arm == "H2":
        base["drop_prob"] = 0.25
        base["protocol"] = str(base["protocol"]) + " H2_drop0.25"
    elif arm == "H3":
        base["patience"] = 40
        base["protocol"] = str(base["protocol"]) + " H3_pat40"
    elif arm == "H4":
        base["lr"] = 3e-4
        base["protocol"] = str(base["protocol"]) + " H4_lr3e-4"
    elif arm == "H5":
        base["grad_accum"] = max(4, int(base["grad_accum"]))
        base["protocol"] = str(base["protocol"]) + " H5_near_s3"
    else:
        raise ValueError(f"未知臂 {arm}")
    if seed is not None:
        base["seed"] = int(seed)
    return SharedTrainHP(**base)


def run_arm(arm: str, max_folds: int, run_tag: str, seed: int | None = None) -> dict:
    hp = _hp_variant(arm, seed=seed)
    torch.set_num_threads(int(hp.torch_num_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(hp.cudnn_benchmark)
        print("device", torch.cuda.get_device_name(0))

    data_dir, prefix = resolve_data(hp.data_tag)
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)

    out_dir = exp35_out() / "H" / arm / f"run_{run_tag}" / "three"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": 35,
        "track": "H",
        "arm": arm,
        "seed": int(hp.seed),
        "hp": asdict(hp),
        "anchor_shallow": ANCHOR_SHALLOW_A59,
        "anchor_conformer": ANCHOR_CONFORMER_A59,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    fold_results = []
    t0 = time.time()
    for info in iter_loso6(subjects):
        if max_folds > 0 and info["fold"] >= max_folds:
            break
        fold_results.append(
            train_one_fold(
                fold_info=info,
                X=X,
                y=y,
                device=device,
                hp=hp,
                out_dir=out_dir,
                model_name="shallow",
                build_model=build_model,
            )
        )
    accs = [float(r["best_val_acc"]) for r in fold_results]
    summary = {
        **meta,
        "n_folds": len(fold_results),
        "val_acc_mean": float(np.mean(accs)) if accs else None,
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "folds": fold_results,
        "elapsed_sec": time.time() - t0,
        "out_dir": str(out_dir),
    }
    if summary["val_acc_mean"] is not None:
        summary["gap_vs_conformer_anchor"] = float(
            ANCHOR_CONFORMER_A59 - summary["val_acc_mean"]
        )
        summary["delta_vs_shallow_anchor"] = float(
            summary["val_acc_mean"] - ANCHOR_SHALLOW_A59
        )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"[{arm}] Val {summary['val_acc_mean']:.4f}±{summary['val_acc_std']:.4f} "
        f"gap_vs_C={summary.get('gap_vs_conformer_anchor')}"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="H0")
    ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 SharedTrainHP.seed")
    args = ap.parse_args()
    tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    arms = [a.strip() for a in args.arm.split(",") if a.strip()]
    all_sum = []
    for arm in arms:
        seed_tag = f"_s{args.seed}" if args.seed is not None else ""
        all_sum.append(run_arm(arm, args.max_folds, f"{tag}_{arm}{seed_tag}", seed=args.seed))

    by = {s["arm"]: s for s in all_sum}
    gate: dict = {"promote_full": [], "hold": []}
    h0 = by.get("H0", {}).get("val_acc_mean")
    if h0 is not None:
        for arm, s in by.items():
            if arm == "H0":
                continue
            m = s.get("val_acc_mean")
            if m is not None and m >= h0 + 0.02:
                gate["promote_full"].append(arm)
            else:
                gate["hold"].append(arm)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "arms": all_sum,
        "fold0_gate": gate,
        "note": "仅对 promote_full 开 --max-folds 0；满折臂建议 ≤2",
    }
    out = exp35_out() / "H" / f"gate_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("gate", gate)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
