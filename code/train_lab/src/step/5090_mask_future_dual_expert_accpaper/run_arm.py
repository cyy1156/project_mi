"""单臂入口 · 5090 掩码未来双专家。

Usage:
  python run_arm.py --arm A0_ref --max-folds 0
  python run_arm.py --arm A0 --max-folds 0
  python run_arm.py --arm P2 --max-folds 1
  python run_arm.py --arm P1 --dry-run
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
from shared_hparams import SHARED  # noqa: E402
from train_kfold import run_pf_kfold  # noqa: E402


def _run_a0_ref(argv_rest: list[str]) -> None:
    """A0-ref：官方 Acc_paper 环 + braindecode Shallow（量级参考）。"""
    from dataclasses import replace as dc_replace

    import torch.nn as nn
    from braindecode.models import ShallowFBCSPNet

    from _paths import OFFICIAL, load_official
    from shared_hparams import OUT_ROOT_TAG

    sys.path.insert(0, str(OFFICIAL))
    import shared_hparams as base_hp  # type: ignore

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
    )
    sys.modules["shared_hparams"] = base_hp

    from _paths import HOP100, PRE, STEP

    for p in (str(STEP), str(PRE), str(HOP100), str(OFFICIAL)):
        if p not in sys.path:
            sys.path.append(p)

    tr = load_official("task_runner")

    def build_model(n_chans, n_times, n_outputs, drop_prob) -> nn.Module:
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            drop_prob=drop_prob,
        )

    # 官方环仍会跑 Task+Three；主报读 three/。方案主表 A0 用自写臂。
    sys.argv = [sys.argv[0], "--data", "openbmi_2s_hop100", *argv_rest]
    tr.run_baseline_main(
        model_name="shallow_A0_ref",
        build_model=build_model,
        input_kind="time",
        structure_note="A0-ref braindecode Shallow · 500pt · Acc_paper（量级参考）",
        extra_meta={"arm": "A0_ref", "package": OUT_ROOT_TAG, "heads": "task+three"},
    )


def main() -> None:
    p = argparse.ArgumentParser(description="5090 mask-future dual-expert arm")
    p.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    p.add_argument("--max-folds", type=int, default=0, help="0=五折全量")
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch-train", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args, rest = p.parse_known_args()

    arm = ARMS[args.arm]
    print(f"[arm] {arm.arm_id}: {arm.note}", flush=True)
    if args.dry_run:
        d = {k: getattr(arm, k) for k in arm.__dataclass_fields__}
        print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
        return

    if arm.arm_id == "A0_ref":
        extra = []
        if args.max_folds:
            extra += ["--max-folds", str(args.max_folds)]
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
    if repl:
        hp = replace(hp, **repl)

    summary = run_pf_kfold(arm, hp=hp, max_folds=args.max_folds)
    print(
        json.dumps(
            {k: summary[k] for k in ("arm", "mean", "std", "run_dir")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
