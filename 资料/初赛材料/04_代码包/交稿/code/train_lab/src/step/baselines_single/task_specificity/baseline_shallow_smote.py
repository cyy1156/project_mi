"""ShallowFBCSPNet：Task-only 五折，SMOTE 对照臂（不改 baseline_shallow / balbatch）。

臂定义（对齐 balbatch 的 B1/B2）：
  S1: 普通 CE + train SMOTE(1:1) + BalAcc 早停
  S2: 加权 CE (w0=2,w1=1) + train SMOTE(1:1) + BalAcc 早停

说明：
  - 仅对每折 train 做 SMOTE；val/test 保持真实比例
  - SMOTE 后 train 用普通 shuffle（不再叠加 WeightedRandomSampler）
  - 仅跑 Task，无 Three

用法（在 task_specificity/ 目录下）：
  python baseline_shallow_smote.py --data merged_2s --arm S1
  python baseline_shallow_smote.py --data merged_2s --arm S2
  python baseline_shallow_smote.py --data merged_2s --arm both
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from braindecode.models import ShallowFBCSPNet

HERE = Path(__file__).resolve().parent  # .../task_specificity
BASELINES_DIR = HERE.parent  # .../baselines_single
STEP_DIR = BASELINES_DIR.parent  # .../step
CODE_ROOT = HERE.parents[4]  # .../code
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
from task_objective import build_task_ce
from task_smote import smote_resample_eeg


@dataclass(frozen=True)
class ArmConfig:
    name: str
    model_name: str
    use_weighted_ce: bool
    task_w0: float
    task_w1: float
    description: str


ARMS: dict[str, ArmConfig] = {
    "S1": ArmConfig(
        name="S1",
        model_name="shallow_smote_balacc",
        use_weighted_ce=False,
        task_w0=1.0,
        task_w1=1.0,
        description="普通CE + train SMOTE(静息对齐任务) + BalAcc早停",
    ),
    "S2": ArmConfig(
        name="S2",
        model_name="shallow_wce2_smote_balacc",
        use_weighted_ce=True,
        task_w0=2.0,
        task_w1=1.0,
        description="加权CE w0=2,w1=1 + train SMOTE + BalAcc早停",
    ),
}

SMOTE_K = 5


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
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


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


def arm_hparams(arm: ArmConfig) -> dict:
    return {
        **shared_as_dict(),
        "arm": arm.name,
        "task_resample": "smote",
        "smote_k_neighbors": SMOTE_K,
        "task_weight_mode": "fixed" if arm.use_weighted_ce else "none",
        "task_w0": arm.task_w0,
        "task_w1": arm.task_w1,
        "task_early_stop": "balanced_accuracy",
        "task_only": True,
    }


def train_task_one_fold(
    fold_info,
    X,
    y,
    subjects,
    device,
    hp: SharedTrainHP,
    out_dir: Path,
    arm: ArmConfig,
) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== TASK [{arm.name}] fold {fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  test ={fold_info['test_subjects']}\n"
        f"  n={int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )

    g = make_generator(hp.seed + fold)
    X_tr_raw = X[masks["train"]]
    y_tr_raw = y[masks["train"]]
    X_tr, y_tr, smote_info = smote_resample_eeg(
        X_tr_raw,
        y_tr_raw,
        random_state=hp.seed + fold,
        k_neighbors=SMOTE_K,
    )
    print(
        f"  SMOTE: n {smote_info['n_before']}→{smote_info['n_after']} | "
        f"n0 {smote_info['n0_before']}→{smote_info['n0_after']} | "
        f"n1 {smote_info['n1_before']}→{smote_info['n1_after']} | "
        f"k={smote_info['k_neighbors']}"
    )

    train_loader = DataLoader(
        ArrayTaskDataset(X_tr, y_tr),
        batch_size=hp.batch_train,
        shuffle=True,
        num_workers=0,
        generator=g,
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

    if arm.use_weighted_ce:
        # 加权基于 SMOTE 后的标签（此时两类已接近平衡；叠 w0=2 可能偏静息）
        criterion = build_task_ce(
            device,
            mode="fixed",
            w0=arm.task_w0,
            w1=arm.task_w1,
            y_train=y_tr,
        )
    else:
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
                    "stage": "task2_shallow_smote",
                    "fold": fold,
                    "model_name": arm.model_name,
                    "arm": arm.name,
                    "n_outputs": 2,
                    "weight_transfer": False,
                    "classifier": "native",
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": jsonify_metrics(m),
                    "smote": smote_info,
                    "hparams": arm_hparams(arm),
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
        "smote": smote_info,
        "test_metrics": m_te,
        "test_metrics_by_dataset": by_ds,
    }


def run_task_kfold(
    X, y, subjects, device, hp: SharedTrainHP, out_dir: Path, data_tag: str, arm: ArmConfig
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for info in iter_folds(subjects, hp, data_tag):
        folds.append(train_task_one_fold(info, X, y, subjects, device, hp, out_dir, arm))

    val_bal = [r["best_val_balanced_accuracy"] for r in folds]
    val_f1s = [r["best_val_f1"] for r in folds]
    test_spec = [r["test_metrics"]["specificity"] for r in folds]
    test_rec = [r["test_metrics"]["recall"] for r in folds]
    test_bal = [r["test_metrics"]["balanced_accuracy"] for r in folds]
    test_f1s = [r["test_metrics"]["f1"] for r in folds]
    test_accs = [r["test_metrics"]["accuracy"] for r in folds]

    def ms(xs):
        return _mean_std(xs)

    print(
        f"\n[TASK {arm.name}] Val BalAcc {ms(val_bal)[0]:.4f}±{ms(val_bal)[1]:.4f} | "
        f"Test Spec {ms(test_spec)[0]:.4f}±{ms(test_spec)[1]:.4f} | "
        f"Test Rec {ms(test_rec)[0]:.4f}±{ms(test_rec)[1]:.4f} | "
        f"Test BalAcc {ms(test_bal)[0]:.4f}±{ms(test_bal)[1]:.4f} | "
        f"Test F1 {ms(test_f1s)[0]:.4f}±{ms(test_f1s)[1]:.4f}"
    )
    summary = {
        "task": "task_kfold",
        "model_name": arm.model_name,
        "arm": arm.name,
        "data_tag": data_tag,
        "hparams": arm_hparams(arm),
        "shallow": {"backbone": "ShallowFBCSPNet"},
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
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def run_one_arm(arm: ArmConfig, data_tag: str, X, y_task, subjects, device, hp: SharedTrainHP) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = TRAIN_LAB / "out" / "baseline" / arm.model_name / data_tag / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    records_root = REPO_ROOT / "资料" / "模型训练"
    run_md_dir = records_root / "runs" / f"{stamp}_{arm.model_name}"
    md_path = run_md_dir / f"{arm.model_name}五折实验记录.md"
    data_dir, prefix = resolve_data(data_tag)

    append_md(
        md_path,
        "\n".join(
            [
                f"# 被试独立五折实验记录（{stamp} / {arm.model_name}）",
                "",
                f"- 开始：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- data：`{data_dir}`（prefix=`{prefix}`）",
                f"- model：`{arm.model_name}`（新脚本 baseline_shallow_smote.py；不改 baseline_shallow.py）",
                f"- 臂：`{arm.name}` — {arm.description}",
                f"- Task 重采样：仅 train 折 SMOTE（k={SMOTE_K}，展平 8×T）；val/test 真实比例",
                f"- 早停：Balanced Acc；仅跑 Task（无 Three）",
                f"- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）",
                f"- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）",
                f"- shared hp：`{shared_as_dict()}`",
                f"- weight_transfer：`False` | classifier：`native`",
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
        f"start arm={arm.name} model={arm.model_name} data={data_tag} device={device}",
    )

    sum_task = run_task_kfold(
        X, y_task, subjects, device, hp, out_root / "task", data_tag, arm
    )
    log_line(
        log_path,
        f"TASK done arm={arm.name} val_BalAcc={sum_task['val_balanced_accuracy_mean']:.4f} "
        f"test_Spec={sum_task['test_specificity_mean']:.4f} "
        f"test_Rec={sum_task['test_recall_mean']:.4f} "
        f"test_BalAcc={sum_task['test_balanced_accuracy_mean']:.4f} "
        f"test_F1={sum_task['test_f1_mean']:.4f}",
    )

    append_md(
        md_path,
        "\n".join(
            [
                "## 最终结论",
                "",
                f"### Task（静息/任务）— 臂 {arm.name}",
                f"- Val BalAcc（选模）：`{sum_task['val_balanced_accuracy_mean']:.4f} ± {sum_task['val_balanced_accuracy_std']:.4f}`",
                f"- Test Spec：`{sum_task['test_specificity_mean']:.4f} ± {sum_task['test_specificity_std']:.4f}`",
                f"- Test Rec：`{sum_task['test_recall_mean']:.4f} ± {sum_task['test_recall_std']:.4f}`",
                f"- Test BalAcc：`{sum_task['test_balanced_accuracy_mean']:.4f} ± {sum_task['test_balanced_accuracy_std']:.4f}`",
                f"- Test F1（附报）：`{sum_task['test_f1_mean']:.4f} ± {sum_task['test_f1_std']:.4f}`",
                f"- Test Acc：`{sum_task['test_acc_mean']:.4f} ± {sum_task['test_acc_std']:.4f}`",
                "",
                *task_fold_md_lines(sum_task["folds"]),
                "### 实验超参",
                "```json",
                json.dumps(arm_hparams(arm), indent=2),
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
        "model_name": arm.model_name,
        "arm": arm.name,
        "data_tag": data_tag,
        "stamp": stamp,
        "weight_transfer": False,
        "classifier": "native",
        "task_only": True,
        "task": sum_task,
        "md": str(md_path),
        "out_root": str(out_root),
    }
    (out_root / "final_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log_line(log_path, f"done arm={arm.name} md={md_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shallow SMOTE 对照：仅 Task；臂 S1/S2（对齐 balbatch 结构）"
    )
    p.add_argument("--data", default=SHARED.data_tag, help="merged_2s | bci2a_2s | stieger_2s")
    p.add_argument(
        "--arm",
        default="S1",
        choices=("S1", "S2", "both"),
        help="S1=普通CE+SMOTE；S2=w0=2加权CE+SMOTE；both=依次跑两臂",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm_names = ("S1", "S2") if args.arm == "both" else (args.arm,)
    for name in arm_names:
        run_one_arm(ARMS[name], data_tag, X, y_task, subjects, device, hp)


if __name__ == "__main__":
    main()
