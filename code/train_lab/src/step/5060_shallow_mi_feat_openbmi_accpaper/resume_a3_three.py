"""恢复 A3 Three：重评已完成 fold0/1，从 fold2 起重训 fold2–4（降内存）。"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
CODE_ROOT = HERE.parents[3]  # .../code
PRE_ROOT = CODE_ROOT / "preprocess_lab"
for p in (HERE, STEP, PRE_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from channel_fe import prepare_laterality_mu_X
from shared_hparams import SHARED, OUT_ROOT_TAG
from perf_loader import apply_runtime_threads
from task_runner import (
    REPO_ROOT,
    TRAIN_LAB,
    _eval_split,
    _mean_std,
    append_md,
    log_line,
    seed_everything,
    train_one_fold,
)
from data_paths import resolve_data
from src.common.steps.split_subjects import iter_subject_kfold
from md_fold_detail import three_fold_md_lines

RUN = (
    TRAIN_LAB
    / "out"
    / OUT_ROOT_TAG
    / "shallow_a3_lat_mu_openbmi_2s_hop100_balbatch_accpaper"
    / "openbmi_2s_hop100"
    / "run_20260809_064233"
)
MD = (
    REPO_ROOT
    / "资料"
    / "模型训练"
    / "runs"
    / "5060_shallow_mi_feat"
    / "20260809_064233_shallow_a3_lat_mu_openbmi_2s_hop100_balbatch_accpaper"
    / "shallow_a3_lat_mu_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md"
)


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


def reeval_fold(fold_info, X, y, subjects, trial_ids, device, hp, x_path, ckpt: Path) -> dict:
    fold = fold_info["fold"]
    n_times = int(X.shape[-1])
    from task_runner import _n_chans_of

    n_chans = _n_chans_of(X)
    model = build_model(n_chans, n_times, 3, hp.drop_prob).to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    te_trial, te_win, _ = _eval_split(
        model,
        X,
        y,
        subjects,
        trial_ids,
        fold_info["masks"]["test"],
        device,
        hp,
        input_kind="time",
        n_classes=3,
        x_path=x_path,
    )
    va_trial, va_win, va_loss = _eval_split(
        model,
        X,
        y,
        subjects,
        trial_ids,
        fold_info["masks"]["val"],
        device,
        hp,
        input_kind="time",
        n_classes=3,
        x_path=x_path,
    )
    return {
        "fold": fold,
        "best_val_acc_paper": float(va_trial["acc_paper"]),
        "best_val_balacc_maj": float(va_trial["balanced_accuracy"]),
        "best_val_loss": float(va_loss),
        "best_epoch": int(blob.get("epoch", -1)),
        "stopped_epoch": int(blob.get("epoch", -1)),
        "best_val_trial_metrics": va_trial,
        "test_trial_metrics": te_trial,
        "test_window_metrics": te_win,
        "resumed_reeval": True,
    }


def main() -> None:
    # 降内存：0 worker，避免 OpenBLAS/多进程再炸
    hp = replace(
        SHARED,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        batch_train=96,
        batch_eval=192,
    )
    apply_runtime_threads(hp.torch_num_threads)
    seed_everything(hp.seed, cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic)

    data_dir, prefix = resolve_data("openbmi_2s_hop100")
    X0 = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    print("[resume] prepare laterality_mu …", flush=True)
    X = prepare_laterality_mu_X(X0)
    x_path = str(getattr(X, "filename", None) or "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_three = RUN / "three"
    log_path = RUN / "run.log"
    folds_all = list(
        iter_subject_kfold(subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed)
    )

    results: list[dict] = []
    # fold0/1：重评已有 ckpt
    for info in folds_all[:2]:
        ckpt = out_three / f"fold{info['fold']}" / "best_three.pt"
        assert ckpt.is_file(), ckpt
        print(f"[resume] reeval three fold{info['fold']}", flush=True)
        results.append(
            reeval_fold(info, X, y_three, subjects, trial_ids, device, hp, x_path, ckpt)
        )
        print(
            f"  → test Acc_paper={results[-1]['test_trial_metrics']['acc_paper']:.4f}",
            flush=True,
        )

    # fold2–4：清掉不完整 fold2 缓存后重训
    for info in folds_all[2:]:
        fdir = out_three / f"fold{info['fold']}"
        if fdir.is_dir():
            shutil.rmtree(fdir, ignore_errors=True)
        print(f"[resume] train three fold{info['fold']}", flush=True)
        results.append(
            train_one_fold(
                info,
                X,
                y_three,
                subjects,
                trial_ids,
                device,
                hp,
                out_three,
                model_name="shallow_a3_lat_mu_openbmi_2s_hop100_balbatch_accpaper",
                build_model=build_model,
                input_kind="time",
                n_outputs=3,
                ckpt_name="best_three.pt",
                stage_tag="three3_shallow_a3_resume",
                x_path=x_path or None,
            )
        )

    val_ap = [r["best_val_acc_paper"] for r in results]
    test_ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in results]
    test_bm = [float(r["test_trial_metrics"]["balanced_accuracy"]) for r in results]
    test_wbal = [float(r["test_window_metrics"]["balanced_accuracy"]) for r in results]
    summary = {
        "task": "three_kfold_accpaper",
        "model_name": "shallow_a3_lat_mu_openbmi_2s_hop100_balbatch_accpaper",
        "data_tag": "openbmi_2s_hop100",
        "protocol": hp.protocol,
        "resumed": True,
        "val_acc_paper_mean": _mean_std(val_ap)[0],
        "val_acc_paper_std": _mean_std(val_ap)[1],
        "test_acc_paper_mean": _mean_std(test_ap)[0],
        "test_acc_paper_std": _mean_std(test_ap)[1],
        "test_balacc_maj_mean": _mean_std(test_bm)[0],
        "test_balacc_maj_std": _mean_std(test_bm)[1],
        "test_window_balacc_mean": _mean_std(test_wbal)[0],
        "test_window_balacc_std": _mean_std(test_wbal)[1],
        "folds": results,
        "out_dir": str(out_three),
        "hparams": asdict(hp),
    }
    (out_three / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    task_sum = json.loads((RUN / "task" / "summary.json").read_text(encoding="utf-8"))
    meta = {
        "model_name": "shallow_a3_lat_mu_openbmi_2s_hop100_balbatch_accpaper",
        "resumed_three": True,
        "task": task_sum,
        "three": summary,
    }
    (RUN / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    md_tail = [
        "",
        "### Three（resume 完成后补写）",
        f"- Val Acc_paper：`{summary['val_acc_paper_mean']:.4f} ± {summary['val_acc_paper_std']:.4f}`",
        f"- Test Acc_paper：`{summary['test_acc_paper_mean']:.4f} ± {summary['test_acc_paper_std']:.4f}`",
        f"- Test BalAcc_maj：`{summary['test_balacc_maj_mean']:.4f} ± {summary['test_balacc_maj_std']:.4f}`",
        f"- Test 窗级 BalAcc（附报）：`{summary['test_window_balacc_mean']:.4f} ± {summary['test_window_balacc_std']:.4f}`",
        f"- resume：`{datetime.now().isoformat(timespec='seconds')}` · fold0/1 reeval · fold2–4 retrain · num_workers=0",
        "",
        *three_fold_md_lines(results),
    ]
    append_md(MD, "\n".join(md_tail), RUN, log_path)
    log_line(
        log_path,
        f"THREE resume done val_AccPaper={summary['val_acc_paper_mean']:.4f} "
        f"test_AccPaper={summary['test_acc_paper_mean']:.4f}",
    )
    print(
        f"[done] Three Test Acc_paper "
        f"{summary['test_acc_paper_mean']:.4f}±{summary['test_acc_paper_std']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
