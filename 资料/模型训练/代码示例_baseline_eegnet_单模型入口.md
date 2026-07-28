# 代码示例：EEGNet 单模型入口（完整可粘贴）

> 性质：**示例文档**（先给完整代码，便于你手写落地；本步**不要求**已存在对应 `.py`）  
> 目标目录：`code/train_lab/src/step/baselines_single/`  
> 策略：[`资料/实验结果说明/训练策略_二分类与三分类独立训练.md`](../实验结果说明/训练策略_二分类与三分类独立训练.md)  
> 协议：[`正式评估协议_被试独立五折.md`](./正式评估协议_被试独立五折.md)  
> 归类：[`归类清单_单模型入口重构.md`](./归类清单_单模型入口重构.md)

---

## 0. 约定（写代码前先对齐）

| 项 | 约定 |
|----|------|
| 入口 | **一模型一脚本**：本示例为 `baseline_eegnet.py` |
| 超参 | 全基线共用 `shared_hparams.py`；EEGNet **结构参数**（F1/D/F2）写在本脚本 |
| 数据 | 默认 `merged_2s`；也可用 `bci2a_2s` / `stieger_2s`（`data_paths.resolve_data`） |
| 张量 | 盘上 `(N,1,8,500)` → Dataset 变 `(B,8,500)` |
| Task | `n_outputs=2`，Val **F1** 早停；Test 终评 |
| Three | **重新随机初始化**，`n_outputs=3`，Val **F1-macro** 早停；**不加载** Task 权重 |
| 记录 | 每次实验一个时间目录：`资料/模型训练/runs/{stamp}_eegnet/五折实验记录.md`；并更新上级「最新」指针 |
| 权重 | `code/train_lab/out/baseline/eegnet/<data>/run_<stamp>/` |
| 复用 | 上级 `dataset.py` / `metrics.py` / `data_paths.py`（**不要** import 归档里的 registry） |

目录关系：

```text
code/train_lab/src/step/
  dataset.py / metrics.py / data_paths.py     ← 复用
  baselines_single/
    shared_hparams.py                         ← 本文 §1
    baseline_eegnet.py                        ← 本文 §2（完整）
  归档_旧训练入口/                             ← 仅对照，勿当作入口
```

运行（建议）：

```text
cd code/train_lab/src/step/baselines_single
python baseline_eegnet.py
python baseline_eegnet.py --data merged_2s
```

PyCharm：Working directory 设为 `.../src/step/baselines_single`。

### 记录目录约定（按时间命名）

每次跑完，MD **不要**直接堆在 `资料/模型训练/` 根下，而是：

```text
资料/模型训练/
  五折实验记录_最新.md              ← 只做入口指针
  runs/
    {stamp}_{model}/                    ← 例：20260728_180000_eegnet
      五折实验记录.md
```

对应代码（已写入下文 `main` / `append_md`）：

```python
records_root = REPO_ROOT / "资料" / "模型训练"
run_md_dir = records_root / "runs" / f"{stamp}_{MODEL_NAME}"
md_path = run_md_dir / "五折实验记录.md"
```

权重仍在：`code/train_lab/out/baseline/eegnet/<data>/run_{stamp}/`。

---

## 1. `shared_hparams.py`（全基线共用）

路径：`code/train_lab/src/step/baselines_single/shared_hparams.py`

```python
"""全基线共用训练超参（改这里 = 所有 baseline_*.py 一起变）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "merged_2s"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 100
    patience: int = 18
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 7e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.55  # 无 dropout 的模型可忽略


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
```

> 说明：本示例用**单点超参**（不跑网格）。若以后要「全基线同一网格」，可在此文件增加 `GRID = [SharedTrainHP(...), ...]`，各脚本 `for hp in GRID`。

---

## 2. `baseline_eegnet.py`（完整单文件）

路径：`code/train_lab/src/step/baselines_single/baseline_eegnet.py`

下面为一份**可直接粘贴**的完整脚本：Task 五折 → Three 五折（独立初始化）→ 写 MD / `final_meta.json`。

```python
"""EEGNet 单模型入口：Task + Three 独立五折，写 MD。不用 registry。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# braindecode 0.8 兼容：无 EEGNet 时用 EEGNetv4
try:
    from braindecode.models import EEGNet
except ImportError:
    from braindecode.models import EEGNetv4 as EEGNet

# ----- 路径：本文件必须在 step/baselines_single/ -----
# 注意：是 parents（复数），不是 parent[i]
# HERE.parents: [0]=step [1]=src [2]=train_lab [3]=code [4]=MI(仓库根)
HERE = Path(__file__).resolve().parent   # .../baselines_single
STEP_DIR = HERE.parent                   # .../src/step
CODE_ROOT = HERE.parents[3]              # .../code  （不是 parents[4]=MI）
TRAIN_LAB = CODE_ROOT / "train_lab"      # .../code/train_lab
REPO_ROOT = CODE_ROOT.parent             # .../MI
PRE_ROOT = CODE_ROOT / "preprocess_lab"  # .../code/preprocess_lab

# 运行时把上级 step/、preprocess_lab/ 加入模块搜索路径
# （data_paths / dataset / metrics 都在 step/ 下，不在 baselines_single/ 里）
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))
if str(PRE_ROOT) not in sys.path:
    sys.path.insert(0, str(PRE_ROOT))

# 同目录：shared_hparams.py（注意文件名是 shared_，不是 share_）
from shared_hparams import SHARED, SharedTrainHP, shared_as_dict

# 上级 step/ 模块（完整 import，不要写成半截 from dataset）
from data_paths import resolve_data
from dataset import ArrayTaskDataset, ArrayThreeDataset
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

MODEL_NAME = "eegnet"
# EEGNet 结构（不进 shared；其它基线脚本各自写死自己的结构）
EEGNET_F1, EEGNET_D, EEGNET_F2 = 8, 2, 16
```

> **若 PyCharm 仍对 `data_paths` / `resolve_data` 报「未解析的引用」**：  
> 这是 IDE 静态检查问题（它看不懂运行时的 `sys.path.insert`），**不一定表示运行会挂**。  
> 处理：右键 `code/train_lab/src/step` → **将目录标记为 → 源代码根目录（Sources Root）**，再同步一遍。  
> 或：运行配置里把 Working directory 设为 `baselines_single`，Interpreter 用仓库根 `.venv`。

> **对照截图常见笔误（请勿照抄错误写法）**：

| 错误 | 正确 |
|------|------|
| `Path(...).resovle()` | `Path(...).resolve()` |
| `HERE.parent[3]`（单数 parent 不能下标） | `HERE.parents[3]` |
| `HERE.parents[4]` 当作 `code`（实际是仓库根 `MI`） | `HERE.parents[3]` → `code`；`CODE_ROOT.parent` → `MI` |
| `from share_hparams import ...` | `from shared_hparams import ...` |
| `from datasets`（半截，会触发「应为 import」） | `from dataset import ArrayTaskDataset, ArrayThreeDataset` |
| `from data_paths import resolve_data` 写在路径设置之前 | 先 `sys.path.insert`，再 import |

下面从「模型构造」起接完整脚本其余部分（与上一版相同逻辑）：

```python
# ========================= 模型 =========================

def build_eegnet(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    """本脚本内手写构造；原生分类头。"""
    return EEGNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        F1=EEGNET_F1,
        D=EEGNET_D,
        F2=EEGNET_F2,
        drop_prob=drop_prob,
    )


# ========================= 记录 =========================

def append_md(md_path: Path, text: str, out_root: Path, log_path: Path) -> None:
    """md_path 已在「按时间命名」的子目录里；最新指针仍写在 资料/模型训练/ 根下。"""
    records_root = REPO_ROOT / "资料" / "模型训练"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    # 相对 资料/模型训练/ 的链接，例如 runs/20260728_180000_eegnet/五折实验记录.md
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


# ========================= 训练工具 =========================

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


# ========================= Task 五折 =========================

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

    def loader(mask, train: bool):
        return DataLoader(
            ArrayTaskDataset(X[mask], y[mask]),
            batch_size=hp.batch_train if train else hp.batch_eval,
            shuffle=train,
            num_workers=0,
        )

    train_loader = loader(masks["train"], True)
    val_loader = loader(masks["val"], False)
    test_loader = loader(masks["test"], False)

    model = build_eegnet(8, int(X.shape[-1]), 2, hp.drop_prob).to(device)
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
                    "stage": "task2_eegnet",
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "n_outputs": 2,
                    "weight_transfer": False,
                    "classifier": "native",
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
        "eegnet": {"F1": EEGNET_F1, "D": EEGNET_D, "F2": EEGNET_F2},
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


# ========================= Three 五折（独立初始化） =========================

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

    def loader(mask, train: bool):
        return DataLoader(
            ArrayThreeDataset(X[mask], y[mask]),
            batch_size=hp.batch_train if train else hp.batch_eval,
            shuffle=train,
            num_workers=0,
        )

    train_loader = loader(masks["train"], True)
    val_loader = loader(masks["val"], False)
    test_loader = loader(masks["test"], False)

    # 关键：重新 build，不 load Task 的 best_task.pt
    model = build_eegnet(8, int(X.shape[-1]), 3, hp.drop_prob).to(device)
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
                    "stage": "three3_eegnet",
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "n_outputs": 3,
                    "weight_transfer": False,
                    "classifier": "native",
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
        "eegnet": {"F1": EEGNET_F1, "D": EEGNET_D, "F2": EEGNET_F2},
        "val_f1_macro_mean": vm,
        "val_f1_macro_std": vs,
        "test_f1_macro_mean": tm,
        "test_f1_macro_std": ts,
        "test_acc_mean": am,
        "test_acc_std": astd,
        "folds": folds,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\n[THREE] Val F1m {vm:.4f}±{vs:.4f} | Test F1m {tm:.4f}±{ts:.4f}")
    return summary


# ========================= main =========================

def main() -> None:
    p = argparse.ArgumentParser(description="EEGNet 单模型：Task+Three 独立五折")
    p.add_argument("--data", default=SHARED.data_tag, help="merged_2s | bci2a_2s | stieger_2s")
    args = p.parse_args()

    hp = SHARED
    data_tag = args.data
    data_dir, prefix = resolve_data(data_tag)

    X = np.load(data_dir / f"{prefix}_X.npy")
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    assert len(X) == len(y_task) == len(y_three) == len(subjects)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = TRAIN_LAB / "out" / "baseline" / MODEL_NAME / data_tag / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    # 记录根目录 + 每次实验按时间命名的子目录（不要写成 .../"模型训练"/""）
    records_root = REPO_ROOT / "资料" / "模型训练"
    run_md_dir = records_root / "runs" / f"{stamp}_{MODEL_NAME}"
    # 例：资料/模型训练/runs/20260728_180000_eegnet/五折实验记录.md
    md_path = run_md_dir / "五折实验记录.md"

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
                f"- 结构：F1={EEGNET_F1}, D={EEGNET_D}, F2={EEGNET_F2}",
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
    log_line(log_path, f"start model={MODEL_NAME} data={data_tag} device={device}")

    sum_task = run_task_kfold(
        X, y_task, subjects, device, hp, out_root / "task", data_tag
    )
    log_line(
        log_path,
        f"TASK done val_F1={sum_task['val_f1_mean']:.4f} test_F1={sum_task['test_f1_mean']:.4f}",
    )

    sum_three = run_three_kfold(
        X, y_three, subjects, device, hp, out_root / "three", data_tag
    )
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
                "### Three（空闲/左/右，独立初始化）",
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

## 3. 自检清单（粘贴后）

- [ ] `baselines_single/shared_hparams.py` 与 `baseline_eegnet.py` 已按上文创建  
- [ ] 工作目录为 `baselines_single/`（或保证能 `import shared_hparams`）  
- [ ] 已有 `preprocess_lab/out/<data>/{prefix}_X.npy` 等四件套  
- [ ] Three 阶段**没有** `load_state_dict(task_ckpt)`  
- [ ] 报告含 `weight_transfer=False`、`classifier=native`  
- [ ] 未 `from models import build_model`（不用归档 registry）

---

## 4. 与其它基线的关系

复制本脚本为 `baseline_shallow.py` 等时，通常只改：

1. `MODEL_NAME`  
2. `build_*` 函数（换 `ShallowFBCSPNet` / `Deep4Net` / …）  
3. MD / 输出目录里的模型名  

**训练超参继续只读 `SHARED`。**

对照旧实现（勿当入口）：`code/train_lab/src/step/归档_旧训练入口/train_*_kfold.py`。

---

## 5. 一句话

> 在 `baselines_single/` 放 `shared_hparams.py` + `baseline_eegnet.py`：手写 `EEGNet(...)`，共用超参，Task→Three 独立五折，写 MD；不走 registry，不迁权重。
