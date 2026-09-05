"""DBN + 原始时域 + TemporalEncoder：Task / Three 独立五折。

形状约定：
  盘上 X_raw: (N,1,8,500)
  整理后 X:   (N,8,500)
  batch:      (B,8,500)
  Encoder 出: (B,8,D)     D 默认 64
  DBN 入:     flatten (B, 8*D)

落地路径：
  code/train_lab/src/step/Self_development_model/TeporalEncoder_dbn.py
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
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
STEP_DIR = HERE.parent
CODE_ROOT = HERE.parents[3]  # .../code
TRAIN_LAB = CODE_ROOT / "train_lab"
REPO_ROOT = CODE_ROOT.parent  # .../MI
PRE_ROOT = CODE_ROOT / "preprocess_lab"
BASELINES_DIR = STEP_DIR / "baselines_single"

# Self_development_model 下运行时，需同时能 import step/ 与 baselines_single/
for p in (STEP_DIR, BASELINES_DIR, PRE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_hparams import SHARED, SharedTrainHP, shared_as_dict
from data_paths import resolve_data
from metrics import (
    binary_task_metrics,
    format_task_metrics,
    format_three_metrics,
    jsonify_metrics,
    metrics_by_dataset_prefix,
    three_class_metrics,
)
from src.common.steps.split_subjects import (
    iter_subject_kfold,
    iter_subject_kfold_stratified_by_dataset,
)
from md_fold_detail import task_fold_md_lines, three_fold_md_lines

MODEL_NAME = "dbn_raw"
NODE_DIM = 64  # TemporalEncoder 输出 D；DBN 可见层 = 8*D


# ---------------------------------------------------------------------------
# 1) 原始数据整理：不做 bandpower
# ---------------------------------------------------------------------------

def squeeze_raw(X: np.ndarray) -> np.ndarray:
    """(N,1,8,500) 或 (N,8,500) -> (N,8,500) float32。"""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    assert X.ndim == 3 and X.shape[1] == 8, X.shape
    assert X.shape[2] == 500, X.shape  # 2s@250Hz
    return X


# ---------------------------------------------------------------------------
# 2) 时间编码器：每个电极沿时间做共享 1D 卷积，压成 D 维
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """(B, 8, T) -> (B, 8, D)"""

    def __init__(self, n_times: int = 500, node_dim: int = 64, drop_prob: float = 0.5):
        super().__init__()
        self.n_times = n_times
        self.node_dim = node_dim
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Conv1d(32, node_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(node_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # (B*8, D, 1) —— 压的是时间维，不是把 D 变成 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        assert c == 8 and t == self.n_times, x.shape
        x = x.reshape(b * c, 1, t)
        x = self.net(x).squeeze(-1)
        return x.reshape(b, c, self.node_dim)


# ---------------------------------------------------------------------------
# 3) Dataset / 锁种 / DBN（in_channels=D，先经 TemporalEncoder）
# ---------------------------------------------------------------------------

class ArrayFeatDataset(Dataset):
    """(N, 8, F) + 标签；本脚本 F=500（原始时域）。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        assert X.ndim == 3 and X.shape[1] == 8, X.shape
        self.X = X
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


# --- DBN / RBM（摘自 LODO62；监督 forward 不做 RBM 预训练） ---
class RBM(nn.Module):
    def __init__(self, visible_units: int, hidden_units: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))

    def sample_h(self, v: torch.Tensor):
        prob_hidden = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return prob_hidden, torch.bernoulli(prob_hidden)

    def sample_v(self, h: torch.Tensor):
        prob_visible = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return prob_visible, torch.bernoulli(prob_visible)


class DBN(nn.Module):
    """输入 (B, electrodes, in_channels)；内部展平为 electrodes*in_channels。"""

    def __init__(
        self,
        num_electrodes: int = 8,
        in_channels: int = 64,
        num_classes: int = 2,
        hidden_size1: int = 300,
        hidden_size2: int = 400,
    ):
        super().__init__()
        self.rbm1 = RBM(num_electrodes * in_channels, hidden_size1)
        self.rbm2 = RBM(hidden_size1, hidden_size2)
        self.fc = nn.Linear(hidden_size2, num_classes)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # 监督通路：仅用 RBM 权重做 sigmoid 映射 + 线性头；无 CD/预训练步骤
        v = v.view(v.shape[0], -1)  # (B, electrodes*feats)
        h1_prob = torch.sigmoid(torch.matmul(v, self.rbm1.W) + self.rbm1.h_bias)
        h2_prob = torch.sigmoid(torch.matmul(h1_prob, self.rbm2.W) + self.rbm2.h_bias)
        return self.fc(h2_prob)


class DBNRaw(nn.Module):
    """(B,8,500) → TemporalEncoder → DBN → logits"""

    def __init__(
        self,
        n_times: int = 500,
        node_dim: int = 64,
        n_outputs: int = 2,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(n_times=n_times, node_dim=node_dim, drop_prob=drop_prob)
        # DBN 本体无 Dropout；drop_prob 仅用于 Encoder
        self.dbn = DBN(
            num_electrodes=8,
            in_channels=node_dim,
            num_classes=n_outputs,
            hidden_size1=300,
            hidden_size2=400,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dbn(self.encoder(x))


def build_model_raw(
    n_outputs: int,
    drop_prob: float,
    node_dim: int = NODE_DIM,
) -> nn.Module:
    return DBNRaw(
        n_times=500,
        node_dim=node_dim,
        n_outputs=n_outputs,
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


def train_task_one_fold(
    fold_info, X, y, subjects, device, hp: SharedTrainHP, out_dir: Path
) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== TASK fold {fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  test ={fold_info['test_subjects']}\n"
        f"  n={int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )

    g = make_generator(hp.seed + fold)

    def loader(mask, train: bool):
        return DataLoader(
            ArrayFeatDataset(X[mask], y[mask]),
            batch_size=hp.batch_train if train else hp.batch_eval,
            shuffle=train,
            num_workers=0,
            generator=g if train else None,
        )

    train_loader = loader(masks["train"], True)
    val_loader = loader(masks["val"], False)
    test_loader = loader(masks["test"], False)

    seed_everything(hp.seed + fold)
    model = build_model_raw(n_outputs=2, drop_prob=hp.drop_prob, node_dim=NODE_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    best_score, best_state, best_ep, best_val_loss = -1.0, None, 0, float("inf")
    bad, ep = 0, 0
    for ep in range(1, hp.max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, False)
        yt, yp = collect_preds(model, val_loader, device)
        m = binary_task_metrics(yt, yp)
        print(f"fold{fold} ep {ep:03d}  tr={tr:.4f}  va={va:.4f}  val_F1={m['f1']:.4f}")
        if m["f1"] > best_score:
            best_score, best_ep, best_val_loss = m["f1"], ep, va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save(
                {
                    "stage": "task2_dbn_raw",
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "n_outputs": 2,
                    "weight_transfer": False,
                    "classifier": "native",
                    "input": "raw_temporal_encoder",
                    "node_dim": NODE_DIM,
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
    by_ds = metrics_by_dataset_prefix(y_te, p_te, subjects[masks["test"]], binary_task_metrics)
    m_te = by_ds["overall"]
    print(format_task_metrics(f"fold{fold}/test", m_te))
    return {
        "fold": fold,
        "best_val_f1": float(best_score),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "test_metrics": m_te,
        "test_metrics_by_dataset": by_ds,
    }


def run_task_kfold(X, y, subjects, device, hp: SharedTrainHP, out_dir: Path, data_tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for info in iter_folds(subjects, hp, data_tag):
        folds.append(train_task_one_fold(info, X, y, subjects, device, hp, out_dir))
    val_f1s = [r["best_val_f1"] for r in folds]
    test_f1s = [r["test_metrics"]["f1"] for r in folds]
    test_accs = [r["test_metrics"]["accuracy"] for r in folds]
    vm, vs = _mean_std(val_f1s)
    tm, ts = _mean_std(test_f1s)
    am, astd = _mean_std(test_accs)
    summary = {
        "task": "task_kfold",
        "model_name": MODEL_NAME,
        "data_tag": data_tag,
        "hparams": shared_as_dict(),
        "dbn": {"backbone": "DBN", "hidden": [300, 400]},
        "val_f1_mean": vm,
        "val_f1_std": vs,
        "test_f1_mean": tm,
        "test_f1_std": ts,
        "test_acc_mean": am,
        "test_acc_std": astd,
        "folds": folds,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\n[TASK] Val F1 {vm:.4f}±{vs:.4f} | Test F1 {tm:.4f}±{ts:.4f}")
    return summary


def train_three_one_fold(
    fold_info, X, y, subjects, device, hp: SharedTrainHP, out_dir: Path
) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== THREE fold {fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  test ={fold_info['test_subjects']}"
    )

    g = make_generator(hp.seed + fold)

    def loader(mask, train: bool):
        return DataLoader(
            ArrayFeatDataset(X[mask], y[mask]),
            batch_size=hp.batch_train if train else hp.batch_eval,
            shuffle=train,
            num_workers=0,
            generator=g if train else None,
        )

    train_loader = loader(masks["train"], True)
    val_loader = loader(masks["val"], False)
    test_loader = loader(masks["test"], False)

    seed_everything(hp.seed + fold)
    model = build_model_raw(n_outputs=3, drop_prob=hp.drop_prob, node_dim=NODE_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    best_score, best_state, best_ep, best_val_loss = -1.0, None, 0, float("inf")
    bad, ep = 0, 0
    for ep in range(1, hp.max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, False)
        yt, yp = collect_preds(model, val_loader, device)
        m = three_class_metrics(yt, yp)
        print(
            f"fold{fold} ep {ep:03d}  tr={tr:.4f}  va={va:.4f}  "
            f"val_F1m={m['f1_macro']:.4f}"
        )
        if m["f1_macro"] > best_score:
            best_score, best_ep, best_val_loss = m["f1_macro"], ep, va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save(
                {
                    "stage": "three3_dbn_raw",
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "n_outputs": 3,
                    "weight_transfer": False,
                    "classifier": "native",
                    "input": "raw_temporal_encoder",
                    "node_dim": NODE_DIM,
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": jsonify_metrics(m),
                    "hparams": shared_as_dict(),
                },
                fold_dir / "best_three.pt",
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  early stop @ ep {ep}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    y_te, p_te = collect_preds(model, test_loader, device)
    by_ds = metrics_by_dataset_prefix(y_te, p_te, subjects[masks["test"]], three_class_metrics)
    m_te = by_ds["overall"]
    print(format_three_metrics(f"fold{fold}/test", m_te))
    return {
        "fold": fold,
        "best_val_f1_macro": float(best_score),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "test_metrics": m_te,
        "test_metrics_by_dataset": by_ds,
    }


def run_three_kfold(X, y, subjects, device, hp: SharedTrainHP, out_dir: Path, data_tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for info in iter_folds(subjects, hp, data_tag):
        folds.append(train_three_one_fold(info, X, y, subjects, device, hp, out_dir))
    val_f1s = [r["best_val_f1_macro"] for r in folds]
    test_f1s = [r["test_metrics"]["f1_macro"] for r in folds]
    test_accs = [r["test_metrics"]["accuracy"] for r in folds]
    vm, vs = _mean_std(val_f1s)
    tm, ts = _mean_std(test_f1s)
    am, astd = _mean_std(test_accs)
    summary = {
        "task": "three_kfold",
        "model_name": MODEL_NAME,
        "data_tag": data_tag,
        "weight_transfer": False,
        "hparams": shared_as_dict(),
        "dbn": {"backbone": "DBN", "hidden": [300, 400]},
        "val_f1_macro_mean": vm,
        "val_f1_macro_std": vs,
        "test_f1_macro_mean": tm,
        "test_f1_macro_std": ts,
        "folds": folds,
        "out_dir": str(out_dir),
        "test_acc_mean": am,
        "test_acc_std": astd,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\n[THREE] Val F1m {vm:.4f}±{vs:.4f} | Test F1m {tm:.4f}±{ts:.4f}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="DBN 单模型：特征 + Task/Three 独立五折")
    p.add_argument("--data", default=SHARED.data_tag, help="merged_2s | bci2a_2s | stieger_2s")
    args = p.parse_args()

    hp = SHARED
    seed_everything(hp.seed)
    data_tag = args.data
    data_dir, prefix = resolve_data(data_tag)

    X_raw = np.load(data_dir / f"{prefix}_X.npy")
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    assert len(X_raw) == len(y_task) == len(y_three) == len(subjects)

    # 时域 -> 特征立方体 (N,8,2)；模型吃 (B,8,F)
    X = squeeze_raw(X_raw)
    assert X.shape == (len(X_raw), 8, 500), X.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = TRAIN_LAB / "out" / "baseline" / MODEL_NAME / data_tag / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    records_root = REPO_ROOT / "资料" / "模型训练"
    run_md_dir = records_root / "runs" / f"{stamp}_{MODEL_NAME}"
    md_path = run_md_dir / f"{MODEL_NAME}五折实验记录.md"

    append_md(
        md_path,
        "\n".join(
            [
                f"# 被试独立五折实验记录（{stamp} / {MODEL_NAME}）",
                "",
                f"- 开始：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- data：`{data_dir}`（prefix=`{prefix}`）",
                f"- model：`{MODEL_NAME}`（单脚本；无 registry）",
                f"- 输入：原始时域 `{X.shape}` → TemporalEncoder(D={NODE_DIM}) → DBN",
                f"- 结构：Encoder + DBN(in={NODE_DIM}, hidden 300/400)；监督 forward，无 RBM 预训练",
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
    log_line(log_path, f"start model={MODEL_NAME} data={data_tag} device={device} X={X.shape}")

    sum_task = run_task_kfold(X, y_task, subjects, device, hp, out_root / "task", data_tag)
    log_line(
        log_path,
        f"TASK done val_F1={sum_task['val_f1_mean']:.4f} test_F1={sum_task['test_f1_mean']:.4f}",
    )

    sum_three = run_three_kfold(X, y_three, subjects, device, hp, out_root / "three", data_tag)
    log_line(
        log_path,
        f"THREE done val_F1m={sum_three['val_f1_macro_mean']:.4f} "
        f"test_F1m={sum_three['test_f1_macro_mean']:.4f}",
    )

    append_md(
        md_path,
        "\n".join(
            [
                "## 最终结论",
                "",
                "### Task（静息/任务）",
                f"- Val F1：`{sum_task['val_f1_mean']:.4f} ± {sum_task['val_f1_std']:.4f}`",
                f"- Test F1：`{sum_task['test_f1_mean']:.4f} ± {sum_task['test_f1_std']:.4f}`",
                f"- Test Acc：`{sum_task['test_acc_mean']:.4f} ± {sum_task['test_acc_std']:.4f}`",
                "",
                *task_fold_md_lines(sum_task["folds"]),
                "### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）",
                f"- Val F1-macro：`{sum_three['val_f1_macro_mean']:.4f} ± {sum_three['val_f1_macro_std']:.4f}`",
                f"- Test F1-macro：`{sum_three['test_f1_macro_mean']:.4f} ± {sum_three['test_f1_macro_std']:.4f}`",
                f"- Test Acc：`{sum_three['test_acc_mean']:.4f} ± {sum_three['test_acc_std']:.4f}`",
                "",
                *three_fold_md_lines(sum_three["folds"]),
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
        "model_name": MODEL_NAME,
        "data_tag": data_tag,
        "stamp": stamp,
        "weight_transfer": False,
        "classifier": "native",
        "input": "raw_temporal_encoder",
        "node_dim": NODE_DIM,
        "X_feat_shape": list(X.shape),
        "task": sum_task,
        "three": sum_three,
        "md": str(md_path),
        "out_root": str(out_root),
    }
    (out_root / "final_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log_line(log_path, f"done md={md_path}")


if __name__ == "__main__":
    main()
