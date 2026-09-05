"""DGCNN 单模型入口：特征立方体 + Task/Three 独立五折，写 MD。不用 registry。"""

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
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
STEP_DIR = HERE.parent
CODE_ROOT = HERE.parents[3]  # .../code（parents[4] 是仓库根 MI）
TRAIN_LAB = CODE_ROOT / "train_lab"
REPO_ROOT = CODE_ROOT.parent  # .../MI
PRE_ROOT = CODE_ROOT / "preprocess_lab"

if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))
if str(PRE_ROOT) not in sys.path:
    sys.path.insert(0, str(PRE_ROOT))

from shared_hparams import SHARED, SharedTrainHP, shared_as_dict
from md_fold_detail import task_fold_md_lines, three_fold_md_lines
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

MODEL_NAME = "dgcnn"

# 2 频带 log 功率（对齐预处理 8–30）：(N,1,8,500)@250Hz -> (N,8,2)
BANDS_HZ = ((8.0, 13.0), (13.0, 30.0))  # μ, β

def raw_to_bandpower(X: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    """盘上时域 (N,1,8,500) 或 (N,8,500) -> 特征立方体 (N,8,F) log 功率；F=len(BANDS_HZ)。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    assert X.ndim == 3 and X.shape[1] == 8, X.shape
    n, n_ch, n_times = X.shape
    nyq = sfreq / 2.0
    out = np.empty((n, n_ch, len(BANDS_HZ)), dtype=np.float32)
    for bi, (lo, hi) in enumerate(BANDS_HZ):
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        # filtfilt along time; vectorize over trials*channels
        flat = X.reshape(-1, n_times)
        filt = np.asarray([filtfilt(b, a, row) for row in flat], dtype=np.float64)
        power = np.mean(filt ** 2, axis=1).reshape(n, n_ch)
        out[:, :, bi] = np.log(power + 1e-10).astype(np.float32)
    return out


class ArrayFeatDataset(Dataset):
    """五折用：特征立方体 (N, 8, F)，标签 Task 或 Three。"""

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

# --- DGCNN（摘自 LODO62；默认改 8 导联） ---
def laplacian(w: torch.Tensor) -> torch.Tensor:
    d = torch.sum(w, dim=1)
    d_re = 1 / torch.sqrt(d + 1e-5)
    d_matrix = torch.diag_embed(d_re)
    lap = torch.eye(d_matrix.shape[0], device=w.device) - torch.matmul(torch.matmul(d_matrix, w), d_matrix)
    return lap


class GraphConv(nn.Module):
    def __init__(self, k: int, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def chebyshev_polynomial(self, x: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        if self.k < 1:
            return torch.empty(0, device=x.device)
        t_k = []
        if self.k >= 1:
            t_k.append(x)
        if self.k >= 2:
            t_k.append(torch.matmul(lap, x))
        for _ in range(2, self.k):
            t_k.append(2 * torch.matmul(lap, t_k[-1]) - t_k[-2])
        return torch.stack(t_k, dim=1)

    def forward(self, x: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        cp = self.chebyshev_polynomial(x, lap)
        cp = cp.permute(0, 2, 3, 1).flatten(start_dim=2)
        return torch.matmul(cp, self.weight)


class B1ReLU(nn.Module):
    def __init__(self, bias_shape: int):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(1, 1, bias_shape))
        self.relu = nn.ReLU()
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bias + x)


class B2ReLU(nn.Module):
    def __init__(self, bias_shape1: int, bias_shape2: int):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(1, bias_shape1, bias_shape2))
        self.relu = nn.ReLU()
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bias + x)


class DGCNN(nn.Module):
    def __init__(
        self,
        num_electrodes: int = 8,
        in_channels: int = 2,
        num_classes: int = 2,
        k: int = 2,
        relu_is: int = 1,
        layers: list[int] | None = None,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        if layers is None:
            layers = [128]
        self.dropout_rate = dropout_rate
        self.layers = layers
        self.k = k
        self.num_electrodes = num_electrodes
        self.relu_is = relu_is

        self.graphConvs = nn.ModuleList()
        self.graphConvs.append(GraphConv(self.k, in_channels, self.layers[0]))
        for i in range(len(self.layers) - 1):
            self.graphConvs.append(GraphConv(self.k, self.layers[i], self.layers[i + 1]))

        self.fc = nn.Linear(self.num_electrodes * self.layers[-1], 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)
        self.adj = nn.Parameter(torch.Tensor(self.num_electrodes, self.num_electrodes))
        self.adj_bias = nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU(inplace=True)
        self.b_relus = nn.ModuleList()
        if self.relu_is == 1:
            for i in range(len(self.layers)):
                self.b_relus.append(B1ReLU(self.layers[i]))
        elif self.relu_is == 2:
            for i in range(len(self.layers)):
                self.b_relus.append(B2ReLU(self.adj.shape[0], self.layers[i]))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weight()

    def init_weight(self) -> None:
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0, std=0.1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, electrodes, feats)
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)
        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.b_relus[i](x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_model(n_electrodes: int, n_feats: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return DGCNN(
        num_electrodes=n_electrodes,
        in_channels=n_feats,
        num_classes=n_outputs,
        k=2,
        layers=[128],
        dropout_rate=drop_prob,
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
    n_feats = int(X.shape[-1])
    model = build_model(8, n_feats, 2, hp.drop_prob).to(device)
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
                    "stage": "task2_dgcnn",
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "n_outputs": 2,
                    "weight_transfer": False,
                    "classifier": "native",
                    "input": "bandpower_cube",
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
        "dgcnn": {"backbone": "DGCNN", "k": 2, "layers": [128]},
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
    n_feats = int(X.shape[-1])
    model = build_model(8, n_feats, 3, hp.drop_prob).to(device)
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
                    "stage": "three3_dgcnn",
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "n_outputs": 3,
                    "weight_transfer": False,
                    "classifier": "native",
                    "input": "bandpower_cube",
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
        "dgcnn": {"backbone": "DGCNN", "k": 2, "layers": [128]},
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
    p = argparse.ArgumentParser(description="DGCNN 单模型：特征 + Task/Three 独立五折")
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
    X = raw_to_bandpower(X_raw, sfreq=250.0)
    assert X.ndim == 3 and X.shape[1] == 8 and X.shape[2] == len(BANDS_HZ), X.shape

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
                f"- 输入：bandpower 立方体 `{X.shape}`（非时域 500）",
                f"- 结构：DGCNN(k=2, layers=[128], dropout=shared drop_prob)",
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
        "input": "bandpower_cube",
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
