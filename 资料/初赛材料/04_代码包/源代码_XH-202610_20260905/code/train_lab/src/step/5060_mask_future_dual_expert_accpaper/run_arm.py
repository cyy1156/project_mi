"""单臂入口 · 5060 掩码未来双专家（低内存 · 默认 fold0）.

Usage:
  python run_arm.py --arm A0              # 默认 max-folds=1
  python run_arm.py --arm P1 --max-folds 1
  python run_arm.py --arm P2 --max-folds 0   # 五折（建议改到 5090 包）
  python run_arm.py --arm P1 --dry-run

  # 带外部看门狗（推荐 16GB 机）:
  powershell -File run_with_mem_guard.ps1 -Arm P1 -ExtraArgs "--max-folds 1 --num-workers 0"
  powershell -File run_gate_chain_guarded.ps1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms_registry import ARMS  # noqa: E402
from mem_guard import MemGuardLimits, start_mem_guard  # noqa: E402
from shared_hparams import SHARED  # noqa: E402
from train_kfold import run_pf_kfold  # noqa: E402


def _win_commit_limit_gb() -> float:
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPageFile / (1024**3)


def _ensure_mem_guard() -> None:
    commit_limit = _win_commit_limit_gb()
    start_mem_guard(
        MemGuardLimits(
            max_process_virt_gb=40.0,
            max_process_ws_gb=14.0,
            # 与外部看门狗对齐；折间页缓存谷底允许略低于 0.08
            min_sys_free_phys_gb=0.05,
            max_sys_commit_used_gb=60.0,
            max_sys_commit_ratio=0.98,
            allow_pagefile_grow=True,
            poll_sec=1.0,
        )
    )
    print(
        f"[mem_guard] commit_limit_now={commit_limit:.2f}G "
        f"(growable pagefile OK; hard caps proc<=40G / sys<=60G / ratio)",
        flush=True,
    )


def _run_a0_ref(argv_rest: list[str]) -> None:
    """A0-ref：官方 Acc_paper 环 + braindecode Shallow（量级参考）。"""
    import importlib.util
    from dataclasses import replace as dc_replace

    import torch.nn as nn
    from braindecode.models import ShallowFBCSPNet

    from _paths import OFFICIAL, load_official
    from shared_hparams import OUT_ROOT_TAG

    # 本包已加载 shared_hparams（含 data_tag_a0）；官方环需要 baselines 的 SharedTrainHP（data_tag）。
    # 必须按路径强制加载 OFFICIAL/shared_hparams.py，避免 sys.modules 命中本包。
    official_hp_path = Path(OFFICIAL) / "shared_hparams.py"
    spec = importlib.util.spec_from_file_location(
        "shared_hparams_official_a0ref", official_hp_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official shared_hparams: {official_hp_path}")
    base_hp = importlib.util.module_from_spec(spec)
    # dataclass 需要模块先挂到 sys.modules
    sys.modules[spec.name] = base_hp
    spec.loader.exec_module(base_hp)

    base_hp.OUT_ROOT_TAG = OUT_ROOT_TAG
    base_hp.SHARED = dc_replace(
        base_hp.SHARED,
        batch_train=SHARED.batch_train,
        batch_eval=SHARED.batch_eval,
        patience=SHARED.patience,
        num_workers=SHARED.num_workers,
        pin_memory=SHARED.pin_memory,
        use_amp=SHARED.use_amp,
        torch_num_threads=SHARED.torch_num_threads,
        persistent_workers=SHARED.persistent_workers,
        cudnn_benchmark=SHARED.cudnn_benchmark,
        non_blocking=SHARED.non_blocking,
    )
    if not hasattr(base_hp.SHARED, "data_tag"):
        raise RuntimeError("official SharedTrainHP missing data_tag after replace")
    # 覆盖本包已加载的 shared_hparams，供官方 task_runner `from shared_hparams import SHARED`
    sys.modules["shared_hparams"] = base_hp

    from _paths import HOP100, PRE, STEP

    for p in (str(STEP), str(PRE), str(HOP100), str(OFFICIAL)):
        if p not in sys.path:
            sys.path.append(p)
    # OFFICIAL 必须先于本包，避免二次 import 又拾到 scheme17 shared_hparams
    for p in (str(HERE), str(OFFICIAL)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(OFFICIAL))

    # 强制重载官方环（避免缓存到错误 SHARED）
    sys.modules.pop("_mfde5060_official_task_runner", None)
    tr = load_official("task_runner")
    tr.SHARED = base_hp.SHARED
    if hasattr(tr, "SharedTrainHP"):
        tr.SharedTrainHP = base_hp.SharedTrainHP
    if hasattr(tr, "OUT_ROOT_TAG"):
        tr.OUT_ROOT_TAG = OUT_ROOT_TAG

    def build_model(n_chans, n_times, n_outputs, drop_prob) -> nn.Module:
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            drop_prob=drop_prob,
        )

    # 方案17 定稿：仅 Three（不做 Task 五折）
    if "--three-only" not in argv_rest and "--skip-task" not in argv_rest:
        argv_rest = ["--three-only", *argv_rest]
    sys.argv = [sys.argv[0], "--data", "openbmi_2s_hop100", *argv_rest]
    tr.run_baseline_main(
        model_name="shallow_A0_ref",
        build_model=build_model,
        input_kind="time",
        structure_note="A0-ref braindecode Shallow · 500pt · Three-only · Acc_paper（量级参考 · 5060）",
        extra_meta={"arm": "A0_ref", "package": OUT_ROOT_TAG, "heads": "three-only"},
    )


def main() -> None:
    p = argparse.ArgumentParser(description="5060 mask-future dual-expert arm (low-mem)")
    p.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    # 5060 默认 fold0 门控；全量五折请显式 --max-folds 0 或改用 5090 包
    p.add_argument(
        "--max-folds",
        type=int,
        default=1,
        help="默认 1=仅 fold0；0=五折全量（本机慎用）",
    )
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch-train", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-mem-guard", action="store_true", help="关闭进程内内存看门狗")
    p.add_argument(
        "--resume-dir",
        default="",
        help="已有 run 目录：跳过已有 fold*/metrics.json，续跑剩余折并写 summary.json",
    )
    args, rest = p.parse_known_args()

    arm = ARMS[args.arm]
    print(f"[arm] {arm.arm_id}: {arm.note}", flush=True)
    if arm.arm_id.startswith("U"):
        from arms_registry import assert_u_arm_flags

        assert_u_arm_flags()
    if arm.arm_id.startswith("T"):
        from arms_registry import assert_t_arm_flags

        assert_t_arm_flags()
    if args.dry_run:
        d = {k: getattr(arm, k) for k in arm.__dataclass_fields__}
        print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
        return

    if not args.no_mem_guard:
        _ensure_mem_guard()

    if arm.arm_id == "A0_ref":
        extra = []
        # A0_ref 官方环：max_folds 1 需显式传入；0 表示全量
        if args.max_folds > 0:
            extra += ["--max-folds", str(args.max_folds)]
        elif args.max_folds == 0:
            pass  # 官方默认五折
        if args.num_workers >= 0:
            extra += ["--num-workers", str(args.num_workers)]
        if args.max_epochs:
            extra += ["--max-epochs", str(args.max_epochs)]
        if args.patience:
            extra += ["--patience", str(args.patience)]
        if args.batch_train:
            extra += ["--batch-train", str(args.batch_train)]
        _run_a0_ref(extra + rest)
        return

    if arm.arm_id == "L1":
        print(
            "[L1] 请按 实验方案/L1_超参与结构扫描.md 手动网格；自动 chain 已跳过。",
            flush=True,
        )
        return

    hp = SHARED
    repl = {}
    if args.num_workers >= 0:
        repl["num_workers"] = args.num_workers
        repl["persistent_workers"] = args.num_workers > 0
    if args.max_epochs > 0:
        repl["max_epochs"] = args.max_epochs
    if args.patience > 0:
        repl["patience"] = args.patience
    if args.batch_train > 0:
        repl["batch_train"] = args.batch_train
    if "embed_dim" in arm.extra:
        repl["embed_dim"] = int(arm.extra["embed_dim"])
    if "batch_train" in arm.extra and args.batch_train <= 0:
        repl["batch_train"] = int(arm.extra["batch_train"])
    if repl:
        hp = replace(hp, **repl)

    resume = Path(args.resume_dir) if args.resume_dir else None
    summary = run_pf_kfold(
        arm, hp=hp, max_folds=args.max_folds, resume_dir=resume
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("arm", "mean", "std", "run_dir")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
