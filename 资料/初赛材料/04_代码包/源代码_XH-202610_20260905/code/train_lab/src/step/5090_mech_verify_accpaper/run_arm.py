"""5090 · 方案 23 机制验证 · OpenBMI Acc_paper.

设备：NVIDIA RTX 5090 · batch 256/512 · 默认 num_workers=4
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms_registry import ARMS, assert_23_arm_flags  # noqa: E402
from mem_guard import MemGuardLimits, start_mem_guard  # noqa: E402
from shared_hparams import SHARED  # noqa: E402
from train_23_kfold import run_23_kfold  # noqa: E402


def _ensure_mem_guard() -> None:
    start_mem_guard(
        MemGuardLimits(
            max_process_virt_gb=96.0,
            max_process_ws_gb=64.0,
            min_sys_free_phys_gb=0.20,
            max_sys_commit_used_gb=120.0,
            max_sys_commit_ratio=0.98,
            allow_pagefile_grow=True,
            poll_sec=1.0,
        )
    )


def main() -> None:
    assert_23_arm_flags()
    p = argparse.ArgumentParser(description="Scheme-23 mechanism verification (5090)")
    p.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    p.add_argument("--max-folds", type=int, default=0, help="0=五折；1=fold0")
    p.add_argument("--batch-train", type=int, default=0)
    p.add_argument("--batch-eval", type=int, default=0)
    p.add_argument("--resume-dir", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mem-guard", action="store_true", help="启用进程内 mem_guard（默认关）")
    p.add_argument("--num-workers", type=int, default=-1)
    args = p.parse_args()

    arm = ARMS[args.arm]
    hp = SHARED
    if args.batch_train > 0:
        hp = replace(hp, batch_train=args.batch_train)
    if args.batch_eval > 0:
        hp = replace(hp, batch_eval=args.batch_eval)
    if args.num_workers >= 0:
        hp = replace(
            hp,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0,
            persistent_workers=args.num_workers > 0,
        )
    elif sys.platform == "win32":
        # Windows spawn：长跑链脚本显式传 --num-workers 0
        hp = replace(hp, num_workers=0, pin_memory=False, persistent_workers=False)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "arm": arm.arm_id,
                    "note": arm.note,
                    "geom": arm.geom_id,
                    "device": "5090",
                    "batch": [hp.batch_train, hp.batch_eval],
                },
                indent=2,
            )
        )
        return

    if args.mem_guard:
        _ensure_mem_guard()

    resume = Path(args.resume_dir) if args.resume_dir else None
    summary = run_23_kfold(
        arm,
        hp=hp,
        max_folds=args.max_folds,
        resume_dir=resume,
    )
    print(
        f"[{arm.arm_id}] DONE test_acc_paper="
        f"{summary['test_acc_paper_mean']:.4f}±{summary['test_acc_paper_std']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
