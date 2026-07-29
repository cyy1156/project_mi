# 代码示例：DBN 单模型入口（完整可粘贴）

> 性质：**示例文档**，尚未写入仓库 `.py`；按本文手写落地  
> 目标：`code/train_lab/src/step/baselines_single/baseline_dbn.py`  
> 策略：[`资料/实验结果说明/训练策略_二分类与三分类独立训练.md`](../实验结果说明/训练策略_二分类与三分类独立训练.md)  
> 协议：[`正式评估协议_被试独立五折.md`](./正式评估协议_被试独立五折.md)  
> 对照：[`代码示例_baseline_eegnet_单模型入口.md`](./代码示例_baseline_eegnet_单模型入口.md)  
> README：[`baselines_single/README.md`](../../code/train_lab/src/step/baselines_single/README.md)  
> 索引：[`代码示例_图特征基线_DGCNN_GCBNet_DBN.md`](./代码示例_图特征基线_DGCNN_GCBNet_DBN.md)

---

## 0. 关键约定（写代码前先对齐）

| 项 | 约定 |
|----|------|
| 入口 | **一模型一脚本**：落地为 `baseline_dbn.py` |
| 输入 | **特征立方体** `(B, 8, F)`，**不是**时域 `(B, 8, 500)` |
| 特征 | 脚本内 `raw_to_bandpower`：盘上 `(N,1,8,500)@250Hz` → `(N,8,5)`；频带 `(1–4),(4–8),(8–13),(13–30),(30–45)` Hz，取 **log 功率** |
| Dataset | 本脚本内 `ArrayFeatDataset`（接受 `(N,8,F)`）；**不用** `ArrayTaskDataset`（那是时域） |
| 超参 | **复用**已有 `baselines_single/shared_hparams.py`（**不要**再粘贴一份） |
| 复用 | `data_paths` / `metrics` / `split_subjects`（与 shallow/eegnet 相同） |
| Task | `n_outputs=2`，Val **F1** 早停；Test 终评 |
| Three | **重新随机初始化**，`n_outputs=3`，Val **F1-macro** 早停；**不加载** Task 权重 |
| 记录 | `资料/模型训练/runs/{stamp}_dbn/dbn五折实验记录.md` |
| 权重 | `code/train_lab/out/baseline/dbn/<data>/run_<stamp>/` |
| 锁种 | `main`：`seed_everything(hp.seed)`；每折建模型前：`seed_everything(hp.seed+fold)`；Train DataLoader：`generator=make_generator(hp.seed+fold)`，`num_workers=0` |
| 路径 | `CODE_ROOT = HERE.parents[3]`（与 shallow 一致） |

**模型来源**：WeChat LODO `【LODO62】DBN.py` 中的 `RBM` + `DBN`；hidden `300/400`。**注意**：本脚本监督 `forward` **不做** RBM 对比散度预训练，仅用 RBM 权重矩阵做两层 sigmoid 映射 + 线性分类头。`drop_prob` 保留在 `build_model` 签名中以对齐其它基线，但 DBN **不使用** Dropout。

目录关系：

```text
code/train_lab/src/step/
  data_paths.py / metrics.py                  ← 复用
  baselines_single/
    shared_hparams.py                         ← 已有，复用，本文不贴全文
    baseline_dbn.py                                  ← 按本文粘贴落地
```

运行（落地后）：

```text
cd code/train_lab/src/step/baselines_single
python baseline_dbn.py
python baseline_dbn.py --data merged_2s
```

---

## 1. `shared_hparams.py`（复用已有，不粘贴）

路径：`code/train_lab/src/step/baselines_single/shared_hparams.py`

本文脚本只 `from shared_hparams import SHARED, SharedTrainHP, shared_as_dict`。

改训练超参（lr / seed / patience 等）请直接改仓库里那一份；**全基线共用**，勿在本文件再复制一份 dataclass。

---

## 2. `baseline_dbn.py`（完整单文件）

路径：`code/train_lab/src/step/baselines_single/baseline_dbn.py`

下面为一份**可直接粘贴**的完整脚本：在线 bandpower → Task 五折 → Three 五折（独立初始化）→ 写 MD / `final_meta.json`。

```python
"""DBN 单模型入口：特征立方体 + Task/Three 独立五折，写 MD。不用 registry。"""

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

MODEL_NAME = "dbn"

# 5 频带 log 功率：(N,1,8,500)@250Hz -> (N,8,5)
BANDS_HZ = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))

def raw_to_bandpower(X: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    """盘上时域 (N,1,8,500) 或 (N,8,500) -> 特征立方体 (N,8,5) log 功率。"""
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
    def __init__(
        self,
        num_electrodes: int = 8,
        in_channels: int = 5,
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


def build_model(n_electrodes: int, n_feats: int, n_outputs: int, drop_prob: float) -> nn.Module:
    # drop_prob 与其它基线签名对齐；DBN 无 Dropout，此处忽略
    _ = drop_prob
    return DBN(
        num_electrodes=n_electrodes,
        in_channels=n_feats,
        num_classes=n_outputs,
        hidden_size1=300,
        hidden_size2=400,
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
                    "stage": "task2_dbn",
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
                    "stage": "three3_dbn",
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

    # 时域 -> 特征立方体 (N,8,5)；模型吃 (B,8,F)
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
                f"- 结构：DBN(hidden 300/400)；监督 forward，无 RBM 预训练；drop_prob 忽略",
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
                "### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）",
                f"- Val F1-macro：`{sum_three['val_f1_macro_mean']:.4f} ± {sum_three['val_f1_macro_std']:.4f}`",
                f"- Test F1-macro：`{sum_three['test_f1_macro_mean']:.4f} ± {sum_three['test_f1_macro_std']:.4f}`",
                f"- Test Acc：`{sum_three['test_acc_mean']:.4f} ± {sum_three['test_acc_std']:.4f}`",
                "",
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

```

---

## 3. 落地检查清单

- [ ] 已粘贴为 `baselines_single/baseline_dbn.py`（仓库此前**无**此 `.py`）
- [ ] **未**再写一份 `shared_hparams.py`；能 `from shared_hparams import SHARED`
- [ ] 工作目录为 `baselines_single/`（或保证能 import 同级模块）
- [ ] 盘上仍是 `(N,1,8,500)` 时域；`main` 内调用 `raw_to_bandpower` 得到 `(N,8,5)`
- [ ] DataLoader 用 `ArrayFeatDataset`，模型输入 `(B,8,F)` 而非 `(B,8,500)`
- [ ] `CODE_ROOT = HERE.parents[3]`；MD 路径为 `dbn五折实验记录.md`
- [ ] Task → Three **独立**训练；Three **没有** `load_state_dict(task_ckpt)`
- [ ] 每折 `seed_everything(hp.seed+fold)`；Train `generator=make_generator(hp.seed+fold)`；`num_workers=0`
- [ ] 已知：无 RBM 预训练；`drop_prob` 签名保留但未接入网络
- [ ] 未 `from models import build_model`（不用归档 registry）

---

## 4. 一句话

> 按本文手写落地 `baseline_dbn.py`：在线 5 频带 log 功率特征 → DBN → Task/Three 独立五折；复用 `shared_hparams`，不迁权重，不走 registry。
