"""Scheme 16 arms: S0 / H1 / H2 / H3 on braindecode Shallow + hier loss.

Usage:
  python run_arm.py --arm H1 --three-only     # Three fold0 (default for H*)
  python run_arm.py --arm H1 --max-folds 0    # Three 五折
  python run_arm.py --arm S0 --three-only
  python run_arm.py --arm H1 --with-task      # Task(CE)+Three(H1)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import task_runner as tr  # noqa: E402  # sets shared_hparams + official path
from hier_loss import HierLossHP, loss_meta  # noqa: E402
from perf_loader import apply_runtime_threads  # noqa: E402
from raw_time_openbmi import squeeze_raw_2s_openbmi  # noqa: E402
from shared_hparams import OUT_ROOT_TAG, SHARED, shared_as_dict  # noqa: E402
from _official_load import HOP100, OFFICIAL, PRE, STEP  # noqa: E402

for _p in (str(STEP), str(PRE), str(HOP100), str(OFFICIAL)):
    if _p not in sys.path:
        sys.path.append(_p)

ARMS = {
    "S0": dict(note="S0 plain CE · shallow"),
    "H1": dict(note="H1 CE+MI+LR · shallow"),
    "H2": dict(note="H2 H1+margin+idle_suppress · shallow"),
    "H3": dict(note="H3 H2 (+trial_cons later) · shallow"),
}


def build_model(n_chans, n_times, n_outputs, drop_prob) -> nn.Module:
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="16 shallow hier-loss arm")
    p.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    p.add_argument("--with-task", action="store_true", help="Also run Task (CE)")
    p.add_argument("--three-only", action="store_true", default=False)
    args, rest = p.parse_known_args()

    three_only = (not args.with_task) or args.three_only
    if args.arm != "S0" and "--max-folds" not in rest:
        rest = [*rest, "--max-folds", "1"]
    if "--num-workers" not in rest:
        rest = [*rest, "--num-workers", "0"]

    sys.argv = [sys.argv[0], *rest]
    tr.ACTIVE_ARM = args.arm
    meta = loss_meta(args.arm, HierLossHP(arm=args.arm))
    meta["structure_note"] = ARMS[args.arm]["note"]
    meta["prepare_X"] = "squeeze_raw_2s_openbmi"

    if three_only:
        _run_three_only(
            model_name=f"shallow_hier_{args.arm.lower()}",
            build_model=build_model,
            structure_note=ARMS[args.arm]["note"],
            extra_meta=meta,
            argv_rest=rest,
        )
    else:
        tr.run_baseline_main(
            model_name=f"shallow_hier_{args.arm.lower()}",
            build_model=build_model,
            input_kind="time",
            structure_note=ARMS[args.arm]["note"],
            extra_meta=meta,
            # 与 three_only 一致：全库 float16 squeeze，降低折内 pack 峰值
            prepare_X=squeeze_raw_2s_openbmi,
        )


def _run_three_only(
    *,
    model_name: str,
    build_model,
    structure_note: str,
    extra_meta: dict,
    argv_rest: list[str],
) -> None:
    mod = sys.modules["_hier_loss_official_task_runner"]
    from data_paths import resolve_data  # noqa: WPS433
    from src.common.steps.split_subjects import iter_subject_kfold

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=SHARED.data_tag, choices=("openbmi_2s_hop100",))
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=-1)
    ap.add_argument("--batch-train", type=int, default=0)
    ap.add_argument("--batch-eval", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--resume-dir",
        type=str,
        default="",
        help="复用已有 run_*/ 目录（可跳过已存在的 fold pack）",
    )
    args = ap.parse_args(argv_rest)

    hp = SHARED
    repl: dict = {}
    if args.max_epochs > 0:
        repl["max_epochs"] = args.max_epochs
    if args.patience > 0:
        repl["patience"] = args.patience
    if args.num_workers >= 0:
        repl["num_workers"] = args.num_workers
        repl["persistent_workers"] = args.num_workers > 0
    if args.batch_train > 0:
        repl["batch_train"] = args.batch_train
    if args.batch_eval > 0:
        repl["batch_eval"] = args.batch_eval
    if args.no_amp:
        repl["use_amp"] = False
    if args.deterministic:
        repl["deterministic"] = True
        repl["cudnn_benchmark"] = False
    if repl:
        hp = replace(hp, **repl)

    apply_runtime_threads(hp.torch_num_threads)
    mod.seed_everything(
        hp.seed, cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic
    )

    data_tag = args.data
    data_dir, prefix = resolve_data(data_tag)
    x_npy = data_dir / f"{prefix}_X.npy"
    # OpenBMI 时域：先 squeeze 到磁盘 float16 (N,8,500)，再折内 pack。
    # 与正式 shallow 默认直读 float32 源不同，语义等价、显著降低 Windows 文件缓存峰值。
    X = np.load(x_npy, mmap_mode="r")
    print(f"[load] prepare_X squeeze_raw on mmap X{tuple(X.shape)} …", flush=True)
    X = squeeze_raw_2s_openbmi(X)
    fn = getattr(X, "filename", None)
    x_path = str(fn) if fn else str(x_npy)
    x_shape = tuple(X.shape)
    print(f"[load] prepare_X done → X{x_shape} dtype={X.dtype} x_path={x_path}", flush=True)

    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_LAB = Path(__file__).resolve().parents[3]
    out_name = f"{model_name}_openbmi_2s_hop100_balbatch_accpaper"
    if args.resume_dir:
        out_root = Path(args.resume_dir)
        if not out_root.is_dir():
            raise FileNotFoundError(f"--resume-dir not found: {out_root}")
        stamp = out_root.name.replace("run_", "", 1) if out_root.name.startswith("run_") else out_root.name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = TRAIN_LAB / "out" / OUT_ROOT_TAG / out_name / data_tag / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"

    mod.log_line(
        log_path,
        f"start mode=fast model={out_name} data={data_tag} device={device} "
        f"arm={tr.ACTIVE_ARM} three_only=1 prepare_X=squeeze_raw_f16 "
        f"X={x_shape}/{X.dtype} resume={int(bool(args.resume_dir))} "
        f"keep_packs={int(bool(getattr(hp, 'keep_fold_packs', False)))}",
    )
    print(
        f"[perf] device={device} arm={tr.ACTIVE_ARM} three_only workers={hp.num_workers}",
        flush=True,
    )

    global_max_folds = int(args.max_folds)

    def _iter_folds():
        for i, info in enumerate(
            iter_subject_kfold(
                subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
            )
        ):
            if global_max_folds > 0 and i >= global_max_folds:
                break
            yield info

    out_dir = out_root / "three"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    import gc

    # 仅通过 src_box 持有全库 mmap；pack 后由 train_one_fold 清空
    src_box: list = [X]
    X = None  # type: ignore[assignment]
    gc.collect()

    for info in _iter_folds():
        if not src_box:
            src_box.append(np.load(x_path, mmap_mode="r"))
        folds.append(
            tr.train_one_fold(
                info,
                src_box[0],
                y_three,
                subjects,
                trial_ids,
                device,
                hp,
                out_dir,
                model_name=out_name,
                build_model=build_model,
                input_kind="time",
                n_outputs=3,
                ckpt_name="best_three.pt",
                stage_tag=f"three3_{out_name}",
                x_path=x_path,
                src_box=src_box,
            )
        )

    val_ap = [r["best_val_acc_paper"] for r in folds]
    test_ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in folds]
    test_bm = [float(r["test_trial_metrics"]["balanced_accuracy"]) for r in folds]
    test_wbal = [float(r["test_window_metrics"]["balanced_accuracy"]) for r in folds]
    summary = {
        "task": "three_kfold_accpaper",
        "model_name": out_name,
        "data_tag": data_tag,
        "protocol": hp.protocol,
        "no_rap": True,
        "balbatch": True,
        "early_stop": "acc_paper",
        "hparams": asdict(hp),
        "val_acc_paper_mean": mod._mean_std(val_ap)[0],
        "val_acc_paper_std": mod._mean_std(val_ap)[1],
        "test_acc_paper_mean": mod._mean_std(test_ap)[0],
        "test_acc_paper_std": mod._mean_std(test_ap)[1],
        "test_balacc_maj_mean": mod._mean_std(test_bm)[0],
        "test_balacc_maj_std": mod._mean_std(test_bm)[1],
        "test_window_balacc_mean": mod._mean_std(test_wbal)[0],
        "test_window_balacc_std": mod._mean_std(test_wbal)[1],
        "test_f1_macro_maj_mean": mod._mean_std(
            [float(r["test_trial_metrics"]["f1_macro"]) for r in folds]
        )[0],
        "test_f1_macro_maj_std": mod._mean_std(
            [float(r["test_trial_metrics"]["f1_macro"]) for r in folds]
        )[1],
        "folds": folds,
        "out_dir": str(out_dir),
        "max_folds": global_max_folds or hp.n_folds,
        "three_only": True,
    }
    summary.update(extra_meta)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    mod.log_line(
        log_path,
        f"THREE done val_AccPaper={summary['val_acc_paper_mean']:.4f} "
        f"test_AccPaper={summary['test_acc_paper_mean']:.4f}",
    )
    meta = {
        "model_name": out_name,
        "data_tag": data_tag,
        "stamp": stamp,
        "train_device": str(device),
        "three_only": True,
        "shared_hp": shared_as_dict(),
        "three": summary,
        "structure_note": structure_note,
    }
    meta.update(extra_meta)
    (out_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    mod.log_line(log_path, "done")
    print(
        f"[done] Three test Acc_paper={summary['test_acc_paper_mean']:.4f} "
        f"out={out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
