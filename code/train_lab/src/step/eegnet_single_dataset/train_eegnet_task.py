"""单库 EEGNet Task 五折（加权 CE w0=2.2 + Val BalAcc 早停；无 batch balance；完整指标写 summary/MD）。

供 bci2a_2s / bci2a_4s / stieger_2s / stieger_4s 子目录脚本调用；不改 baselines_single/baseline_eegnet.py。
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

PKG_DIR = Path(__file__).resolve().parent  # eegnet_single_dataset
STEP_DIR = PKG_DIR.parent  # step
BASELINES_DIR = STEP_DIR / "baselines_single"
SPECIFICITY_DIR = BASELINES_DIR / "task_specificity"
CODE_ROOT = STEP_DIR.parents[2]  # code
TRAIN_LAB = CODE_ROOT / "train_lab"
REPO_ROOT = CODE_ROOT.parent
PRE_ROOT = CODE_ROOT / "preprocess_lab"

for p in (PKG_DIR, SPECIFICITY_DIR, BASELINES_DIR, STEP_DIR, PRE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_hparams import SHARED, SharedTrainHP, shared_as_dict
from md_fold_detail import task_fold_md_lines
from data_paths import resolve_data
from dataset import ArrayTaskDataset
from metrics import binary_task_metrics, format_task_metrics, jsonify_metrics
from task_objective import build_task_ce
from src.common.steps.split_subjects import iter_subject_kfold

EEGNET_F1, EEGNET_D, EEGNET_F2 = 8, 2, 16
TASK_W0, TASK_W1 = 2.2, 1.0  # 静息 / 任务


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


def build_eegnet(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
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


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def train_task_one_fold(fold_info, X, y, device, hp: SharedTrainHP, out_dir: Path, model_name: str) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== TASK fold {fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  test ={fold_info['test_subjects']}\n"
        f"  n={int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}",
        flush=True,
    )
    g = make_generator(hp.seed + fold)

    def loader(mask, train: bool):
        return DataLoader(
            ArrayTaskDataset(X[mask], y[mask]),
            batch_size=hp.batch_train if train else hp.batch_eval,
            shuffle=train,
            num_workers=0,
            generator=g if train else None,
        )

    train_loader = loader(masks["train"], True)
    val_loader = loader(masks["val"], False)
    test_loader = loader(masks["test"], False)

    seed_everything(hp.seed + fold)
    model = build_eegnet(8, int(X.shape[-1]), 2, hp.drop_prob).to(device)
    criterion = build_task_ce(device, mode="fixed", w0=TASK_W0, w1=TASK_W1)
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
            f"val_F1={m['f1']:.4f} Spec={m['specificity']:.4f} Rec={m['recall']:.4f}",
            flush=True,
        )
        score = float(m["balanced_accuracy"])
        if score > best_score:
            best_score, best_ep, best_val_loss = score, ep, va
            best_val_f1 = float(m["f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save(
                {
                    "stage": "task2_eegnet_single_dataset",
                    "fold": fold,
                    "model_name": model_name,
                    "n_outputs": 2,
                    "weight_transfer": False,
                    "classifier": "native",
                    "task_early_stop": "balanced_accuracy",
                    "task_sampler": "none",
                    "task_ce": {"mode": "fixed", "w0": TASK_W0, "w1": TASK_W1},
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": jsonify_metrics(m),
                    "hparams": shared_as_dict(),
                },
                fold_dir / "best_task.pt",
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  early stop @ ep {ep}", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    y_te, p_te = collect_preds(model, test_loader, device)
    m_te = binary_task_metrics(y_te, p_te)
    print(format_task_metrics(f"fold{fold}/test", m_te), flush=True)
    return {
        "fold": fold,
        "best_val_balanced_accuracy": float(best_score),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "test_metrics": m_te,
    }


def run_task_kfold(
    X, y, subjects, device, hp: SharedTrainHP, out_dir: Path, data_tag: str, model_name: str
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        folds.append(train_task_one_fold(info, X, y, device, hp, out_dir, model_name))

    def ms(key):
        return _mean_std([float(r["test_metrics"][key]) for r in folds])

    val_bal = _mean_std([r["best_val_balanced_accuracy"] for r in folds])
    val_f1 = _mean_std([r["best_val_f1"] for r in folds])
    summary = {
        "task": "task_kfold",
        "model_name": model_name,
        "data_tag": data_tag,
        "task_early_stop": "balanced_accuracy",
        "task_sampler": "none",
        "task_ce": {"mode": "fixed", "w0": TASK_W0, "w1": TASK_W1},
        "hparams": shared_as_dict(),
        "eegnet": {"F1": EEGNET_F1, "D": EEGNET_D, "F2": EEGNET_F2},
        "n_times": int(X.shape[-1]),
        "n_trials": int(len(y)),
        "class_counts": {
            "rest": int(np.sum(y == 0)),
            "task": int(np.sum(y == 1)),
        },
        "val_balanced_accuracy_mean": val_bal[0],
        "val_balanced_accuracy_std": val_bal[1],
        "val_f1_mean": val_f1[0],
        "val_f1_std": val_f1[1],
        "test_acc_mean": ms("accuracy")[0],
        "test_acc_std": ms("accuracy")[1],
        "test_specificity_mean": ms("specificity")[0],
        "test_specificity_std": ms("specificity")[1],
        "test_recall_mean": ms("recall")[0],
        "test_recall_std": ms("recall")[1],
        "test_precision_mean": ms("precision")[0],
        "test_precision_std": ms("precision")[1],
        "test_f1_mean": ms("f1")[0],
        "test_f1_std": ms("f1")[1],
        "test_balanced_accuracy_mean": ms("balanced_accuracy")[0],
        "test_balanced_accuracy_std": ms("balanced_accuracy")[1],
        "folds": folds,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(
        f"\n[TASK] Val BalAcc {summary['val_balanced_accuracy_mean']:.4f}±"
        f"{summary['val_balanced_accuracy_std']:.4f} | "
        f"Test Acc {summary['test_acc_mean']:.4f}±{summary['test_acc_std']:.4f} | "
        f"Spec {summary['test_specificity_mean']:.4f}±{summary['test_specificity_std']:.4f} | "
        f"Rec {summary['test_recall_mean']:.4f}±{summary['test_recall_std']:.4f} | "
        f"Prec {summary['test_precision_mean']:.4f}±{summary['test_precision_std']:.4f} | "
        f"F1 {summary['test_f1_mean']:.4f}±{summary['test_f1_std']:.4f} | "
        f"BalAcc {summary['test_balanced_accuracy_mean']:.4f}±{summary['test_balanced_accuracy_std']:.4f}",
        flush=True,
    )
    return summary


def run_experiment(data_tag: str, model_name: str | None = None, window_note: str = "") -> None:
    hp = SHARED
    seed_everything(hp.seed)
    model_name = model_name or f"eegnet_{data_tag}"
    data_dir, prefix = resolve_data(data_tag)

    X = np.load(data_dir / f"{prefix}_X.npy")
    if X.ndim == 4 and X.shape[1] == 1:
        pass  # (N,1,8,T) ok for ArrayTaskDataset
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    assert len(X) == len(y_task) == len(subjects)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = TRAIN_LAB / "out" / "eegnet_single_dataset" / model_name / data_tag / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    records_root = REPO_ROOT / "资料" / "模型训练"
    md_path = records_root / "runs" / f"{stamp}_{model_name}" / f"{model_name}五折实验记录.md"

    counts = np.bincount(y_task.astype(int), minlength=2)
    append_md(
        md_path,
        "\n".join(
            [
                f"# 被试独立五折实验记录（{stamp} / {model_name}）",
                "",
                f"- 开始：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- data：`{data_dir}`（prefix=`{prefix}`，tag=`{data_tag}`）",
                f"- 切窗说明：{window_note or '见该数据预处理配置'}",
                f"- model：`{model_name}`（EEGNet 加权 CE w0={TASK_W0}/w1={TASK_W1}；**仅 Task**；无 batch balance / 无 Focal）",
                f"- 结构：F1={EEGNET_F1}, D={EEGNET_D}, F2={EEGNET_F2}；n_times=`{int(X.shape[-1])}`",
                f"- 样本：N=`{len(y_task)}` Rest=`{int(counts[0])}` Task=`{int(counts[1])}`",
                f"- 划分：被试独立五折（非 LOSO、非混合库）",
                f"- train sampler：无（普通 shuffle）",
                f"- 损失：固定加权 CE，静息 w0=`{TASK_W0}`，任务 w1=`{TASK_W1}`",
                f"- 早停/选模：Val **Balanced Acc**（附报 Val F1）",
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
        f"start model={model_name} data={data_tag} device={device} N={len(y_task)} "
        f"early_stop=balanced_accuracy sampler=none ce_w0={TASK_W0} ce_w1={TASK_W1}",
    )

    summary = run_task_kfold(
        X, y_task, subjects, device, hp, out_root / "task", data_tag, model_name
    )
    append_md(md_path, "\n".join(task_fold_md_lines(summary["folds"])), out_root, log_path)
    append_md(
        md_path,
        "\n".join(
            [
                "",
                "## 最终结论（Test 五折均值）",
                "",
                f"- Val BalAcc（选模）：`{summary['val_balanced_accuracy_mean']:.4f} ± {summary['val_balanced_accuracy_std']:.4f}`",
                f"- Val F1（附报）：`{summary['val_f1_mean']:.4f} ± {summary['val_f1_std']:.4f}`",
                f"- Test Acc：`{summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}`",
                f"- Test Spec：`{summary['test_specificity_mean']:.4f} ± {summary['test_specificity_std']:.4f}`",
                f"- Test Rec：`{summary['test_recall_mean']:.4f} ± {summary['test_recall_std']:.4f}`",
                f"- Test Precision：`{summary['test_precision_mean']:.4f} ± {summary['test_precision_std']:.4f}`",
                f"- Test F1：`{summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}`",
                f"- Test BalAcc：`{summary['test_balanced_accuracy_mean']:.4f} ± {summary['test_balanced_accuracy_std']:.4f}`",
                "",
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
                "model_name": model_name,
                "data_tag": data_tag,
                "md": str(md_path),
                "out_root": str(out_root),
                "summary": {
                    k: summary[k]
                    for k in summary
                    if k != "folds"
                },
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    log_line(log_path, f"done md={md_path}")


def main_cli(default_data: str, model_name: str, window_note: str) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=default_data)
    args = p.parse_args()
    run_experiment(args.data, model_name=model_name, window_note=window_note)
