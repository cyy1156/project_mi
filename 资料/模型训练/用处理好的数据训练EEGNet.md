# 用处理好的数据训练 EEGNet（入门步骤）

> **【现行入口 · 2026-07 更新】**  
> - 环境：仓库根 `D:\cyy\MI\.venv`  
> - 数据：`code/preprocess_lab/out/bci2a_2s/bci2a_*.npy`，形状 `(N,1,8,500)`  
> - 正式评估（被试独立五折）：  
>   `cd code\train_lab\src\step` →  
>   `python train_task_kfold.py` / `python train_three_kfold.py` / `python run_overnight_kfold.py`  
> - **训练策略（新约定）**：二分类与三分类**独立训练、不迁权重、原生分类头** →  
>   [`资料/实验结果说明/训练策略_二分类与三分类独立训练.md`](../实验结果说明/训练策略_二分类与三分类独立训练.md)  
> - 下文仍含早期 `train_task.py`（试次混合 8:2）教学示例，可作对照；**正式实验以 `*_kfold` 为准**。  
> - 文中若出现 Desktop / `code\.venv` 路径，一律改为 `D:\cyy\MI` / 根 `.venv`。

> **性质**：示例文档。  
> **阶段 A**：braindecode `EEGNet`，`n_outputs=2`（静息/任务）。

---

## 怎么读：路径一律相对仓库根 `MI/`（本机 `D:/cyy/MI/`）

### 与当前工程一致的目录

```text
MI/                                   # D:/cyy/MI
├── .venv/                            ← 唯一虚拟环境
├── DATA/bci2a/
├── code/
│   ├── preprocess_lab/
│   │   ├── config/bci2a_2s.yaml
│   │   ├── src/datasets/bci2a/batch.py
│   │   └── out/bci2a_2s/             ← bci2a_X.npy 等（T=500）
│   └── train_lab/
│       ├── out/
│       └── src/step/
│           ├── dataset.py
│           ├── metrics.py
│           ├── train_task_kfold.py   ← 头1 五折（正式）
│           ├── train_three_kfold.py  ← 头2 五折
│           └── run_overnight_kfold.py
└── 资料/模型训练/
```

### 文件对照表

| 用途 | 路径（相对 `MI/`） |
|------|-------------------|
| 预处理批处理 | `code/preprocess_lab` → `python -m src.datasets.bci2a.batch --cfg config/bci2a_2s.yaml` |
| 五折头1 | `code/train_lab/src/step/train_task_kfold.py` |
| 五折头2 | `code/train_lab/src/step/train_three_kfold.py` |
| 过夜网格 | `code/train_lab/src/step/run_overnight_kfold.py` |
| 输入数据 | `code/preprocess_lab/out/bci2a_2s/bci2a_*.npy` |

运行（仓库根 `.venv`）：

```text
cd D:\cyy\MI\code\train_lab\src\step
D:\cyy\MI\.venv\Scripts\python.exe train_task_kfold.py
```

---

## 0. 阶段目标

| 阶段 | 内容 | 状态 |
|------|------|------|
| A | `EEGNet(n_outputs=2)`：静息(0)/任务(1) | ✅ 五折见 `train_task_kfold.py` |
| B | 第二头三分类 | ✅ 五折见 `train_three_kfold.py` |

```text
预处理 X: (N, 1, 8, 500)     # out/bci2a_2s
模型输入: (B, 8, 500)         ← dataset 里去掉中间的 1
标签:     头1 用 y_task；头2 用 y_three
```

---

## 1. 数据就绪

```text
cd D:\cyy\MI\code\preprocess_lab
$env:PYTHONPATH='.'
D:\cyy\MI\.venv\Scripts\python.exe -m src.datasets.bci2a.batch --cfg config/bci2a_2s.yaml
```

快速自检：

```python
from pathlib import Path
import numpy as np

out = Path(r"D:/cyy/MI/code/preprocess_lab/out/bci2a_2s")
X = np.load(out / "bci2a_X.npy")
yt = np.load(out / "bci2a_y_task.npy")
print(X.shape, np.bincount(yt, minlength=2))
assert X.shape[1:] == (1, 8, 500)
print("data ok")
```

---

## 2. 环境

```text
cd D:\cyy\MI
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# 或 train_lab\requirements_train.txt
```

---

> **以下为早期「试次混合 train/val + `train_task.py`」教学原文（可作对照，非正式入口）**

## 3. 流程

```text
preprocess_lab/out/*.npy
        │
        ▼
src/step/dataset.py     → (B, 8, T), y_task
        │
        ▼
braindecode.EEGNet(n_outputs=2)
        │
        ▼
CrossEntropy + src/step/metrics.py
        │
        ▼
train_lab/out/best_task.pt
```

---

## 4. Dataset（完整示例）

| 项 | 内容 |
|----|------|
| **路径** | `code/train_lab/src/step/dataset.py` |
| **作用** | 读 npy；去掉维 `1`；只返回 `y_task` |

**完整示例内容：**

```python
# 文件: code/train_lab/src/step/dataset.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TaskHeadDataset(Dataset):
    """
    阶段 A：静息(0) / 任务(1)。
    __getitem__ 返回:
      x: (8, T) float32
      y: 标量 long，0 或 1
    """

    def __init__(self, data_dir: str | Path, split: str = "train"):
        """
        data_dir: preprocess_lab/out
        split: "train" 或 "val"
        """
        data_dir = Path(data_dir)
        X = np.load(data_dir / f"{split}_X.npy").astype(np.float32)
        # (N, 1, 8, T) → (N, 8, T)
        if X.ndim == 4 and X.shape[1] == 1:
            X = X[:, 0, :, :]
        assert X.ndim == 3 and X.shape[1] == 8, X.shape  # T = X.shape[-1]
        self.X = X
        self.y_task = np.load(data_dir / f"{split}_y_task.npy").astype(np.int64)
        assert len(self.X) == len(self.y_task)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (8, T)
        y = torch.tensor(self.y_task[idx], dtype=torch.long)
        return x, y
```

**自检要点（已核对）：**

- 文件名请用小写 `dataset.py`（不要用 `DataSet.py`，导入易乱）。  
- 确认已是 `(8, T)` braindecode 输入形状；`T` 与预处理窗长一致。

---

## 5. 模型（写在 train_task.py 内，不单独建 eegnet.py）

```python
from braindecode.models import EEGNet

model = EEGNet(
    n_chans=8,
    n_outputs=2,
    n_times=X.shape[-1],  # 2a 现行 500
    F1=8,
    D=2,
    F2=16,
    drop_prob=0.5,
)
# 输入必须是 (B, 8, T)，不是 (B, 1, 8, T)
```

冒烟：

```python
import torch
from braindecode.models import EEGNet

m = EEGNet(n_chans=8, n_outputs=2, n_times=500, F1=8, D=2, F2=16, drop_prob=0.5)
y = m(torch.randn(4, 8, 500))
assert y.shape == (4, 2)
```

---

## 6. 指标（完整示例）

| 项 | 内容 |
|----|------|
| **路径** | `code/train_lab/src/step/metrics.py` |
| **正类** | `y_task=1` 任务；负类 `0` 静息 |

**完整示例内容（注意：不要对 1D 标签用 `np.argmax`）：**

```python
# 文件: code/train_lab/src/step/metrics.py
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_task_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """正类=1(任务)，负类=0(静息)。"""
    # y_true / y_pred 已是类别编号 (N,)，禁止 np.argmax
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def format_task_metrics(part_name: str, m: dict[str, float]) -> str:
    return "\n".join(
        [
            f"===== [{part_name}] 分类头1（静息=0 / 任务=1）braindecode EEGNet =====",
            f"  混淆矩阵: TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}",
            f"  Accuracy      分类准确率   = {m['accuracy']:.4f}",
            f"  Recall        召回率/灵敏度 = {m['recall']:.4f}",
            f"  Specificity   特异性       = {m['specificity']:.4f}",
            f"  Precision     精确率       = {m['precision']:.4f}",
            f"  F1-score      F1          = {m['f1']:.4f}",
            f"  Balanced Acc  平衡准确率   = {m['balanced_accuracy']:.4f}",
            f"  ※ 本期不计算第二分类头 head_three",
        ]
    )
```

指标含义见：`资料/模型训练/实验结果指标说明.md`。

---

## 7. 训练主程序（完整示例）

| 项 | 内容 |
|----|------|
| **路径** | `code/train_lab/src/step/train_task.py` |
| **ROOT** | 文件在 `.../src/step/` → `parents[3]` = `code/` |

**完整示例内容：**

```python
# 文件: code/train_lab/src/step/train_task.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from torch.utils.data import DataLoader

from src.step.dataset import TaskHeadDataset
from src.step.metrics import binary_task_metrics, format_task_metrics

# 本文件: MI/code/train_lab/src/step/train_task.py
# parents[0]=step → [1]=src → [2]=train_lab → [3]=code
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "preprocess_lab" / "out"
OUT_DIR = ROOT / "train_lab" / "out"


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)  # (B, 2)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("基线: braindecode.EEGNet | 阶段 A: n_outputs=2 | 不训第二头")

    if not (DATA_DIR / "train_X.npy").exists():
        raise FileNotFoundError(
            f"找不到预处理数据: {DATA_DIR}\n请先运行 preprocess_lab 的 pipeline。"
        )

    train_loader = DataLoader(
        TaskHeadDataset(DATA_DIR, "train"),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TaskHeadDataset(DATA_DIR, "val"),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model = EEGNet(
        n_chans=8,
        n_outputs=2,
        n_times=X.shape[-1],  # 2a 现行 500
        F1=8,
        D=2,
        F2=16,
        drop_prob=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_score = -1.0
    best_state = None
    epochs = 50
    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device, False)

        y_true, y_pred = collect_preds(model, val_loader, device)
        m = binary_task_metrics(y_true, y_pred)

        print(f"\nep {ep:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(format_task_metrics("val", m))

        if m["f1"] > best_score:
            best_score = m["f1"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    "stage": "A_braindecode_eegnet_task2",
                    "backend": "braindecode.models.EEGNet",
                    "n_outputs": 2,
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": m,
                },
                OUT_DIR / "best_task.pt",
            )
            print("  ↑ saved", OUT_DIR / "best_task.pt")

    # 用 val F1 最好的权重再报指标（不要用「最后一轮」冒充 best）
    if best_state is not None:
        model.load_state_dict(best_state)

    y_va, p_va = collect_preds(model, val_loader, device)
    print(format_task_metrics("val(best)", binary_task_metrics(y_va, p_va)))
    y_tr, p_tr = collect_preds(model, train_loader, device)
    print(format_task_metrics("train(best)", binary_task_metrics(y_tr, p_tr)))
    print("done. best val F1 =", best_score)


if __name__ == "__main__":
    main()
```

### 怎么运行

```powershell
cd D:\cyy\MI\code\train_lab
$env:PYTHONPATH = "."
D:\cyy\MI\.venv\Scripts\python.exe -m src.step.train_task_kfold
```

> 早期教学用的 `train_task`（试次混合）仅作对照；正式请用上面的 `*_kfold`。

---

## 8. 示例代码核对清单（相对旧版文档已修正）

| 问题 | 错误写法 | 正确写法 |
|------|----------|----------|
| 目录少写 `step` | `src/dataset.py`、`from src.dataset` | `src/step/dataset.py`、`from src.step.dataset` |
| ROOT 层数 | 在 `src/` 下用 `parents[2]` | 在 `src/step/` 下用 **`parents[3]`** → `code/` |
| 运行模块名 | `python -m src.train_task` | `python -m src.step.train_task` |
| 指标 | `np.argmax(y_true)`（会毁标签） | `np.asarray(y_true).astype(int)` |
| 输入形状 | `(B,1,8,T)` 直接进 EEGNet | Dataset 先变成 `(B,8,T)`（2a：T=500） |
| 结束评估 | 用最后一轮权重 | 先 `load_state_dict(best_state)` |
| 文件名 | `DataSet.py` | 推荐 `dataset.py` |

名词（trial/epoch/session）见：`资料/模型训练/名词对照_trial_epoch_session_被试.md`。

---

## 9. 结果怎么看

| 现象 | 可能原因 |
|------|----------|
| Accuracy 高、Specificity 低 | 几乎总猜任务 |
| train ≫ val | 过拟合（单被试常见） |
| 形状报错 | 未 squeeze 成 `(B,8,T)` |
| `ModuleNotFoundError: src` | 未在 `train_lab` 下设 `PYTHONPATH=.` |
| `train_task` 为空/无入口 | 对照 §7 把示例粘进 `src/step/train_task.py` |

二分类随机基线 Accuracy ≈ 0.5（类大致均衡时）。

---

## 10. 动手顺序

1. 确认目录为 `train_lab/src/step/{dataset,metrics,train_task}.py`。  
2. 安装 `requirements_train.txt`（可用清华镜像）。  
3. 自检 npy。  
4. 对照本文粘贴/核对三份源码（尤其 `train_task.py` 与 metrics 的 argmax）。  
5. `python -m src.step.train_task`。  
6. 确认 `train_lab/out/best_task.pt`。

---

## 11. 验收

- [ ] 代码在 `src/step/` 下，导入带 `src.step`  
- [ ] `ROOT = parents[3]` 指向 `code/`  
- [ ] `EEGNet(n_outputs=2)`，输入 `(B,8,T)`，`n_times=T`  
- [ ] metrics **无** `argmax` 误用  
- [ ] 日志有 Accuracy / Recall / Specificity  
- [ ] 保存 `best_task.pt`，结束评估用的是 best 权重  

---

## 12. 一句话

> **数据在 `preprocess_lab/out`；训练代码在 `train_lab/src/step/`；用 `from src.step...` 与 `python -m src.step.train_task`；`parents[3]` 定位到 `code/`；braindecode `EEGNet(n_outputs=2)` 做阶段 A 基线。**
