"""EEGNet：Focal Loss + BalAcc 早停（特异度专项；不改上级 baseline_eegnet.py）。

实验序号 F1：γ=2，α_rest=0.75 / α_task=0.25；仅 Task 五折。
用法（在本目录）：
  python baseline_eegnet_focal_balacc.py --data merged_2s
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from braindecode.models import EEGNet
except ImportError:
    from braindecode.models import EEGNetv4 as EEGNet

HERE = Path(__file__).resolve().parent
BASELINES_DIR = HERE.parent
STEP_DIR = BASELINES_DIR.parent
CODE_ROOT = HERE.parents[4]
TRAIN_LAB = CODE_ROOT / "train_lab"
REPO_ROOT = CODE_ROOT.parent
PRE_ROOT = CODE_ROOT / "preprocess_lab"

for p in (HERE, BASELINES_DIR, STEP_DIR, PRE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_hparams import SHARED, SharedTrainHP, shared_as_dict
from md_fold_detail import task_fold_md_lines
from data_paths import resolve_data
from dataset import ArrayTaskDataset
from metrics import (
    binary_task_metrics,
    format_task_metrics,
    jsonify_metrics,
    metrics_by_dataset_prefix,
)
from src.common.steps.split_subjects import (
    iter_subject_kfold,
    iter_subject_kfold_stratified_by_dataset,
)
from task_objective import build_task_focal

MODEL_NAME = "eegnet_focal_balacc"
EXP_ID = "F1"
EEGNET_F1, EEGNET_D, EEGNET_F2 = 8, 2, 16
FOCAL_GAMMA = 2.0
FOCAL_ALPHA0 = 0.75  # rest
FOCAL_ALPHA1 = 0.25  # task


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return EEGNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        F1=EEGNET_F1,
        D=EEGNET_D,
        F2=EEGNET_F2,
        drop_prob=drop_prob,
    )


def append_md(md_path: Path, text: str, out_root: Path, log_path: Path) -> None:
    records_root = REPO_ROOT / "资料" / "模型训练"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    rel = md_path.relative_to(records_root).as_posix()
    (records_root / "五折实验记录_最新.md").write_text(
        f"# 最新实验入口\n\n本次记录：[`{rel}`](./{rel})\n\n"
        f"权重目录：`{out_root}`\n日志：`{log_path}`\n",
        encoding="utf-8",
    )


def log_line(log_path: Path, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@torch.no_grad()
def collect_preds(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        ps.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            if logits.ndim > 2:
                logits = logits.reshape(logits.shape[0], -1)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def iter_folds(subjects: np.ndarray, hp: SharedTrainHP, data_tag: str):
    if data_tag.startswith("merged"):
        return iter_subject_kfold_stratified_by_dataset(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    return iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    )


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def exp_hparams() -> dict:
    return {
        **shared_as_dict(),
        "exp_id": EXP_ID,
        "task_loss": "focal",
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha0": FOCAL_ALPHA0,
        "focal_alpha1": FOCAL_ALPHA1,
        "task_early_stop": "balanced_accuracy",
        "task_only": True,
    }


def train_task_one_fold(fold_info, X, y, subjects, device, hp, out_dir: Path) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== TASK [F1/focal] fold {fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  test ={fold_info['test_subjects']}\n"
        f"  n={int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )

    g = make_generator(hp.seed + fold)
    train_loader = DataLoader(
        ArrayTaskDataset(X[masks["train"]], y[masks["train"]]),
        batch_size=hp.batch_train,
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    val_loader = DataLoader(
        ArrayTaskDataset(X[masks["val"]], y[masks["val"]]),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ArrayTaskDataset(X[masks["test"]], y[masks["test"]]),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=0,
    )

    seed_everything(hp.seed + fold)
    model = build_model(8, int(X.shape[-1]), 2, hp.drop_prob).to(device)
    criterion = build_task_focal(
        device,
        gamma=FOCAL_GAMMA,
        alpha0=FOCAL_ALPHA0,
        alpha1=FOCAL_ALPHA1,
        use_alpha=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    best_score, best_state, best_ep, best_val_loss = -1.0, None, 0, float("inf")
    best_val_f1 = -1.0
    bad, ep = 0, 0
    for ep in range(1, hp.max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, False)
        yt, yp = collect_preds(model, val_loader, device)
        m = binary_task_metrics(yt, yp)
        print(
            f"fold{fold} ep {ep:03d}  tr={tr:.4f}  va={va:.4f}  "
            f"val_BalAcc={m['balanced_accuracy']:.4f}  "
            f"Spec={m['specificity']:.4f}  Rec={m['recall']:.4f}  F1={m['f1']:.4f}"
        )
        score = float(m["balanced_accuracy"])
        if score > best_score:
            best_score, best_ep, best_val_loss = score, ep, va
            best_val_f1 = float(m["f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save(
                {
                    "stage": "task2_eegnet_focal",
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "exp_id": EXP_ID,
                    "n_outputs": 2,
                    "weight_transfer": False,
                    "classifier": "native",
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": jsonify_metrics(m),
                    "hparams": exp_hparams(),
                },
                fold_dir / "best_task.pt",
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  early stop @ ep {ep}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    y_te, p_te = collect_preds(model, test_loader, device)
    by_ds = metrics_by_dataset_prefix(y_te, p_te, subjects[masks["test"]], binary_task_metrics)
    m_te = by_ds["overall"]
    print(format_task_metrics(f"fold{fold}/test", m_te))
    return {
        "fold": fold,
        "best_val_balanced_accuracy": float(best_score),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "test_metrics": m_te,
        "test_metrics_by_dataset": by_ds,
    }


def run_task_kfold(X, y, subjects, device, hp, out_dir: Path, data_tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = [
        train_task_one_fold(info, X, y, subjects, device, hp, out_dir)
        for info in iter_folds(subjects, hp, data_tag)
    ]

    def ms(xs):
        return _mean_std(xs)

    val_bal = [r["best_val_balanced_accuracy"] for r in folds]
    val_f1s = [r["best_val_f1"] for r in folds]
    test_spec = [r["test_metrics"]["specificity"] for r in folds]
    test_rec = [r["test_metrics"]["recall"] for r in folds]
    test_bal = [r["test_metrics"]["balanced_accuracy"] for r in folds]
    test_f1s = [r["test_metrics"]["f1"] for r in folds]
    test_accs = [r["test_metrics"]["accuracy"] for r in folds]

    print(
        f"\n[TASK F1/focal] Val BalAcc {ms(val_bal)[0]:.4f}±{ms(val_bal)[1]:.4f} | "
        f"Test Spec {ms(test_spec)[0]:.4f}±{ms(test_spec)[1]:.4f} | "
        f"Test Rec {ms(test_rec)[0]:.4f}±{ms(test_rec)[1]:.4f} | "
        f"Test BalAcc {ms(test_bal)[0]:.4f}±{ms(test_bal)[1]:.4f} | "
        f"Test Acc {ms(test_accs)[0]:.4f}±{ms(test_accs)[1]:.4f}"
    )
    summary = {
        "task": "task_kfold",
        "model_name": MODEL_NAME,
        "exp_id": EXP_ID,
        "data_tag": data_tag,
        "hparams": exp_hparams(),
        "eegnet": {"F1": EEGNET_F1, "D": EEGNET_D, "F2": EEGNET_F2},
        "val_balanced_accuracy_mean": ms(val_bal)[0],
        "val_balanced_accuracy_std": ms(val_bal)[1],
        "val_f1_mean": ms(val_f1s)[0],
        "val_f1_std": ms(val_f1s)[1],
        "test_specificity_mean": ms(test_spec)[0],
        "test_specificity_std": ms(test_spec)[1],
        "test_recall_mean": ms(test_rec)[0],
        "test_recall_std": ms(test_rec)[1],
        "test_balanced_accuracy_mean": ms(test_bal)[0],
        "test_balanced_accuracy_std": ms(test_bal)[1],
        "test_f1_mean": ms(test_f1s)[0],
        "test_f1_std": ms(test_f1s)[1],
        "test_acc_mean": ms(test_accs)[0],
        "test_acc_std": ms(test_accs)[1],
        "folds": folds,
        "out_dir": str(out_dir),
    }
    pass_gate = (
        summary["test_specificity_mean"] >= 0.40
        and summary["test_recall_mean"] >= 0.75
        and summary["test_balanced_accuracy_mean"] >= 0.65
    )
    summary["pass_gate"] = bool(pass_gate)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="merged_2s")
    args = p.parse_args()
    hp = SHARED
    seed_everything(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir, prefix = resolve_data(args.data)
    X = np.load(data_dir / f"{prefix}_X.npy")
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    y = np.load(data_dir / f"{prefix}_y_task.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = TRAIN_LAB / "out" / "baseline" / MODEL_NAME / args.data / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    records_root = REPO_ROOT / "资料" / "模型训练"
    md_path = records_root / "runs" / f"{stamp}_{MODEL_NAME}" / f"{MODEL_NAME}五折实验记录.md"

    append_md(
        md_path,
        "\n".join(
            [
                f"# 被试独立五折实验记录（{stamp} / {MODEL_NAME}）",
                "",
                f"- 开始：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- data：`{data_dir}`（prefix=`{prefix}`）",
                f"- model：`{MODEL_NAME}`（脚本 baseline_eegnet_focal_balacc.py；**不改** baseline_eegnet.py）",
                f"- 实验序号：`{EXP_ID}` — Focal Loss γ={FOCAL_GAMMA}, "
                f"α_rest={FOCAL_ALPHA0}, α_task={FOCAL_ALPHA1} + BalAcc 早停",
                f"- 仅跑 Task（无 Three）",
                f"- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1/Acc 仅附报）",
                f"- 对照：同数据上 EEGNet A22（加权CE w0=2.2）",
                f"- shared hp：`{shared_as_dict()}`",
                f"- 权重：`{out_root}`",
                "",
                "---",
                "",
            ]
        ),
        out_root,
        log_path,
    )
    log_line(
        log_path,
        f"start model={MODEL_NAME} exp={EXP_ID} data={args.data} device={device} "
        f"gamma={FOCAL_GAMMA} alpha=({FOCAL_ALPHA0},{FOCAL_ALPHA1})",
    )

    summary = run_task_kfold(X, y, subjects, device, hp, out_root / "task", args.data)
    append_md(md_path, "\n".join(task_fold_md_lines(summary["folds"])), out_root, log_path)
    append_md(
        md_path,
        "\n".join(
            [
                "",
                "## 汇总",
                "",
                f"- Test Acc：`{summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}`",
                f"- Test Spec：`{summary['test_specificity_mean']:.4f} ± {summary['test_specificity_std']:.4f}`",
                f"- Test Rec：`{summary['test_recall_mean']:.4f} ± {summary['test_recall_std']:.4f}`",
                f"- Test BalAcc：`{summary['test_balanced_accuracy_mean']:.4f} ± {summary['test_balanced_accuracy_std']:.4f}`",
                f"- Test F1：`{summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}`",
                f"- 过关：`{'是' if summary['pass_gate'] else '否'}`",
                f"- 结束：`{datetime.now().isoformat(timespec='seconds')}`",
                "",
            ]
        ),
        out_root,
        log_path,
    )
    (out_root / "final_meta.json").write_text(
        json.dumps(
            {
                "stamp": stamp,
                "model_name": MODEL_NAME,
                "exp_id": EXP_ID,
                "data_tag": args.data,
                "out_root": str(out_root),
                "md": str(md_path),
                "pass_gate": summary["pass_gate"],
                "test_specificity_mean": summary["test_specificity_mean"],
                "test_recall_mean": summary["test_recall_mean"],
                "test_balanced_accuracy_mean": summary["test_balanced_accuracy_mean"],
                "test_acc_mean": summary["test_acc_mean"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log_line(
        log_path,
        f"done Spec={summary['test_specificity_mean']:.4f} "
        f"Rec={summary['test_recall_mean']:.4f} "
        f"BalAcc={summary['test_balanced_accuracy_mean']:.4f} "
        f"pass={summary['pass_gate']}",
    )


if __name__ == "__main__":
    main()
