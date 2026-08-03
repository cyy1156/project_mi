"""Task-only 五折：Val BalAcc 早停 + train batch balance；原结构、无 RAP。"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
STEP_DIR = HERE.parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
REPO_ROOT = CODE_ROOT.parent
PRE_ROOT = CODE_ROOT / "preprocess_lab"
OLD_BASELINES = STEP_DIR / "baselines_single"

for p in (HERE, STEP_DIR, PRE_ROOT, OLD_BASELINES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_hparams import SHARED, SharedTrainHP, shared_as_dict
from md_fold_detail import task_fold_md_lines
from task_sampler import make_balanced_sampler
from data_paths import resolve_data
from dataset import ArrayTaskDataset
from metrics import (
    binary_task_metrics,
    format_task_metrics,
    jsonify_metrics,
    metrics_by_dataset_prefix,
)
from src.common.steps.split_subjects import iter_subject_kfold

BuildFn = Callable[..., nn.Module]


class ArrayFeatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        assert len(self.X) == len(self.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


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


def append_md(md_path: Path, text: str, out_root: Path, log_path: Path) -> None:
    records_root = REPO_ROOT / "资料" / "模型训练"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    rel = md_path.relative_to(records_root).as_posix()
    latest = records_root / "五折实验记录_最新.md"
    latest.write_text(
        f"# 最新实验入口\n\n"
        f"本次记录：[`{rel}`](./{rel})\n\n"
        f"权重目录：`{out_root}`\n"
        f"日志：`{log_path}`\n",
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
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def train_task_one_fold(
    fold_info,
    X,
    y,
    subjects,
    device,
    hp: SharedTrainHP,
    out_dir: Path,
    *,
    model_name: str,
    build_model: BuildFn,
    input_kind: str,
) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== TASK [{model_name}] fold {fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  test ={fold_info['test_subjects']}\n"
        f"  n={int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )

    g = make_generator(hp.seed + fold)
    y_tr = y[masks["train"]]

    def make_ds(mask):
        if input_kind == "feat":
            return ArrayFeatDataset(X[mask], y[mask])
        return ArrayTaskDataset(X[mask], y[mask])

    train_loader = DataLoader(
        make_ds(masks["train"]),
        batch_size=hp.batch_train,
        sampler=make_balanced_sampler(y_tr, generator=g),
        num_workers=0,
    )
    val_loader = DataLoader(
        make_ds(masks["val"]), batch_size=hp.batch_eval, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        make_ds(masks["test"]), batch_size=hp.batch_eval, shuffle=False, num_workers=0
    )

    seed_everything(hp.seed + fold)
    if input_kind == "feat":
        model = build_model(8, int(X.shape[-1]), 2, hp.drop_prob).to(device)
    else:
        model = build_model(8, int(X.shape[-1]), 2, hp.drop_prob).to(device)

    criterion = nn.CrossEntropyLoss()
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
                    "stage": f"task2_{model_name}_1s",
                    "fold": fold,
                    "model_name": model_name,
                    "n_outputs": 2,
                    "protocol": hp.protocol,
                    "no_rap": True,
                    "task_early_stop": "balanced_accuracy",
                    "task_sampler": "balanced_invfreq",
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
                print(f"  early stop @ ep {ep}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    y_te, p_te = collect_preds(model, test_loader, device)
    by_ds = metrics_by_dataset_prefix(
        y_te, p_te, subjects[masks["test"]], binary_task_metrics
    )
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


def run_task_kfold(
    X,
    y,
    subjects,
    device,
    hp: SharedTrainHP,
    out_dir: Path,
    data_tag: str,
    *,
    model_name: str,
    build_model: BuildFn,
    input_kind: str,
    extra_meta: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        folds.append(
            train_task_one_fold(
                info,
                X,
                y,
                subjects,
                device,
                hp,
                out_dir,
                model_name=model_name,
                build_model=build_model,
                input_kind=input_kind,
            )
        )

    def ms(xs):
        return _mean_std(xs)

    val_bal = [r["best_val_balanced_accuracy"] for r in folds]
    test_bal = [r["test_metrics"]["balanced_accuracy"] for r in folds]
    test_f1s = [r["test_metrics"]["f1"] for r in folds]
    test_accs = [r["test_metrics"]["accuracy"] for r in folds]
    test_spec = [r["test_metrics"]["specificity"] for r in folds]
    test_rec = [r["test_metrics"]["recall"] for r in folds]

    summary = {
        "task": "task_kfold",
        "model_name": model_name,
        "data_tag": data_tag,
        "protocol": hp.protocol,
        "no_rap": True,
        "task_early_stop": "balanced_accuracy",
        "task_sampler": "balanced_invfreq",
        "hparams": shared_as_dict(),
        "val_balanced_accuracy_mean": ms(val_bal)[0],
        "val_balanced_accuracy_std": ms(val_bal)[1],
        "test_balanced_accuracy_mean": ms(test_bal)[0],
        "test_balanced_accuracy_std": ms(test_bal)[1],
        "test_f1_mean": ms(test_f1s)[0],
        "test_f1_std": ms(test_f1s)[1],
        "test_acc_mean": ms(test_accs)[0],
        "test_acc_std": ms(test_accs)[1],
        "test_specificity_mean": ms(test_spec)[0],
        "test_specificity_std": ms(test_spec)[1],
        "test_recall_mean": ms(test_rec)[0],
        "test_recall_std": ms(test_rec)[1],
        "folds": folds,
        "out_dir": str(out_dir),
    }
    if extra_meta:
        summary.update(extra_meta)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(
        f"\n[TASK {model_name}] Val BalAcc {ms(val_bal)[0]:.4f}±{ms(val_bal)[1]:.4f} | "
        f"Test BalAcc {ms(test_bal)[0]:.4f}±{ms(test_bal)[1]:.4f} | "
        f"Test Spec {ms(test_spec)[0]:.4f}±{ms(test_spec)[1]:.4f} | "
        f"Test F1 {ms(test_f1s)[0]:.4f}±{ms(test_f1s)[1]:.4f}"
    )
    return summary


def run_baseline_main(
    *,
    model_name: str,
    build_model: BuildFn,
    input_kind: str = "time",  # time | feat
    structure_note: str,
    extra_meta: dict | None = None,
    prepare_X: Callable[[np.ndarray], np.ndarray] | None = None,
    default_data: str | None = None,
) -> None:
    import argparse

    p = argparse.ArgumentParser(description=f"{model_name} 1s 离线基线（BalAcc+balbatch，无RAP）")
    p.add_argument(
        "--data",
        default=default_data or SHARED.data_tag,
        choices=("bci2a_1s", "stieger_1s"),
        help="单库 tag；不做 merged",
    )
    args = p.parse_args()

    hp = SHARED
    seed_everything(hp.seed)
    data_tag = args.data
    data_dir, prefix = resolve_data(data_tag)

    X = np.load(data_dir / f"{prefix}_X.npy")
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    assert len(X) == len(y_task) == len(subjects)
    assert int(X.shape[-1]) == hp.n_times_expected, (
        f"期望 n_times={hp.n_times_expected}，got {X.shape}"
    )
    if prepare_X is not None:
        X = prepare_X(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{model_name}_1s_balbatch_balacc"
    out_root = TRAIN_LAB / "out" / "baseline_1s" / out_name / data_tag / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    records_root = REPO_ROOT / "资料" / "模型训练"
    run_md_dir = records_root / "runs" / f"{stamp}_{out_name}"
    md_path = run_md_dir / f"{out_name}五折实验记录.md"

    append_md(
        md_path,
        "\n".join(
            [
                f"# 被试独立五折实验记录（{stamp} / {out_name}）",
                "",
                f"- 开始：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- data：`{data_dir}`（prefix=`{prefix}`，**单库不合并**）",
                f"- protocol：`{hp.protocol}` | **no_rap=True**",
                f"- model：`{model_name}`（原结构）",
                f"- 结构：{structure_note}",
                f"- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1",
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
    log_line(log_path, f"start model={out_name} data={data_tag} device={device}")

    sum_task = run_task_kfold(
        X,
        y_task,
        subjects,
        device,
        hp,
        out_root / "task",
        data_tag,
        model_name=out_name,
        build_model=build_model,
        input_kind=input_kind,
        extra_meta=extra_meta,
    )
    log_line(
        log_path,
        f"TASK done val_BalAcc={sum_task['val_balanced_accuracy_mean']:.4f} "
        f"test_BalAcc={sum_task['test_balanced_accuracy_mean']:.4f}",
    )

    append_md(
        md_path,
        "\n".join(
            [
                "## 最终结论（Task only）",
                "",
                f"- Val BalAcc：`{sum_task['val_balanced_accuracy_mean']:.4f} ± {sum_task['val_balanced_accuracy_std']:.4f}`",
                f"- Test BalAcc：`{sum_task['test_balanced_accuracy_mean']:.4f} ± {sum_task['test_balanced_accuracy_std']:.4f}`",
                f"- Test Spec：`{sum_task['test_specificity_mean']:.4f} ± {sum_task['test_specificity_std']:.4f}`",
                f"- Test Rec：`{sum_task['test_recall_mean']:.4f} ± {sum_task['test_recall_std']:.4f}`",
                f"- Test F1：`{sum_task['test_f1_mean']:.4f} ± {sum_task['test_f1_std']:.4f}`",
                f"- Test Acc：`{sum_task['test_acc_mean']:.4f} ± {sum_task['test_acc_std']:.4f}`",
                "",
                *task_fold_md_lines(sum_task["folds"]),
                "### 共用超参",
                "```json",
                json.dumps(shared_as_dict(), indent=2),
                "```",
                "",
                f"- 结束：`{datetime.now().isoformat(timespec='seconds')}`",
                "",
            ]
        ),
        out_root,
        log_path,
    )

    meta = {
        "model_name": out_name,
        "data_tag": data_tag,
        "stamp": stamp,
        "protocol": hp.protocol,
        "no_rap": True,
        "task": sum_task,
        "md": str(md_path),
        "out_root": str(out_root),
    }
    (out_root / "final_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log_line(log_path, f"done md={md_path}")
