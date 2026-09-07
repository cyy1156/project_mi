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
    import importlib.util
    from dataclasses import replace as dc_replace

    import torch.nn as nn
    from braindecode.models import ShallowFBCSPNet

    from _paths import OFFICIAL
    from shared_hparams import OUT_ROOT_TAG

    spec = importlib.util.spec_from_file_location(
        "_mfde5090_official_shared_hparams", OFFICIAL / "shared_hparams.py"
    )
    assert spec and spec.loader
    base_hp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base_hp
    spec.loader.exec_module(base_hp)

    base_hp.OUT_ROOT_TAG = OUT_ROOT_TAG
    # Windows spawn 无法 pickle 动态加载的官方 task_runner；A0_ref 默认 workers=0
    nw = SHARED.num_workers
    if sys.platform == "win32" and not any(
        a in argv_rest for a in ("--num-workers", "--num_workers")
    ):
        nw = 0
    base_hp.SHARED = dc_replace(
        base_hp.SHARED,
        batch_train=SHARED.batch_train,
        batch_eval=SHARED.batch_eval,
        patience=SHARED.patience,
        num_workers=nw,
        pin_memory=SHARED.pin_memory and nw > 0,
        persistent_workers=SHARED.persistent_workers and nw > 0,
        use_amp=SHARED.use_amp,
        torch_num_threads=SHARED.torch_num_threads,
        cudnn_benchmark=SHARED.cudnn_benchmark,
    )
    sys.modules["shared_hparams"] = base_hp

    import task_runner as tr  # noqa: WPS433 — 本地包装，含 md_fold_detail 等

    def build_model(n_chans, n_times, n_outputs, drop_prob) -> nn.Module:
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            drop_prob=drop_prob,
        )

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
    if arm.arm_id.startswith("U"):
        from arms_registry import assert_u_arm_flags

        assert_u_arm_flags()
    if arm.arm_id.startswith("T"):
        from arms_registry import assert_t_arm_flags

        assert_t_arm_flags()
    if arm.extra.get("scheme21"):
        from arms_registry import assert_21_arm_flags

        assert_21_arm_flags()
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
        repl["pin_memory"] = SHARED.pin_memory and args.num_workers > 0
    elif sys.platform == "win32":
        # memmap + spawn 无法稳定多进程；与 A0_ref 一致默认 workers=0
        repl["num_workers"] = 0
        repl["persistent_workers"] = False
        repl["pin_memory"] = False
    if args.max_epochs > 0:
        repl["max_epochs"] = args.max_epochs
    if args.patience > 0:
        repl["patience"] = args.patience
    if args.batch_train > 0:
        repl["batch_train"] = args.batch_train
    bt_arm = int(arm.extra.get("batch_train", 0))
    if bt_arm > 0 and args.batch_train <= 0:
        repl["batch_train"] = bt_arm
    if repl:
        hp = replace(hp, **repl)

    if arm.extra.get("scheme21"):
        from train_21_kfold import run_21_kfold

        summary = run_21_kfold(arm, hp=hp, max_folds=args.max_folds)
    else:
        summary = run_pf_kfold(arm, hp=hp, max_folds=args.max_folds)
    print(
        json.dumps(
            {
                k: summary.get(k)
                for k in (
                    "arm",
                    "test_acc_paper_mean",
                    "test_acc_paper_std",
                    "test_balacc_maj_mean",
                    "test_window_f1_mean",
                    "run_dir",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
