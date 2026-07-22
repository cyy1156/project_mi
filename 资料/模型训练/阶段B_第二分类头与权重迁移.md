# 阶段 B：第二分类头 + 保存/加载第一头权重

> **性质**：示例文档。标明**放哪个文件、完整内容、怎么改、怎么跑**。  
> **约定**：训练步骤可对照实现；**正式报数划分**见下文「评估协议」链接（文档已定，代码尚未切换）。  
> **前提**：阶段 A 已能跑通，且会写出 `code/train_lab/out/best_task.pt`。

---

## 评估协议（正式报数，先读）

阶段 A / B **正式结果**必须按被试独立协议，不要用当前「试次混合 8:2」冒充跨人泛化：

→ **`资料/模型训练/正式评估协议_被试独立五折.md`**

要点摘要：

```text
全体被试 → 固定 seed 均分成 5 组（seed 只为复现，不是多种子代替五折）
每一折：一组 = test（约 20% 人）
        剩余约 80% 人 → 再按人抽 20% = val，其余 = train
        例 100 人：train 64 / val 16 / test 20
训练：可看 train 精度（通常很高）；用 val 的 F1（头2 用 F1-macro）早停
终评：该折结束后 test 只测一次；五折报 mean±std
```

**现状**：仓库里仍是预处理阶段的试次混合 train/val；下面第 1–11 节是「先把第二头跑通」的工程步骤，指标仅作迁移基线。

---

## 0. 先弄清「第二头」在本期怎么做

你的数据本来就是**双标签**（同一份 `X`）：

| 标签文件 | 含义 | 取值 |
|----------|------|------|
| `*_y_task.npy` | 头1：静息 / 任务 | `0` / `1` |
| `*_y_three.npy` | 头2：空闲 / 左 / 右 | `0` / `1` / `2` |

映射回顾：

| 样本 | `y_task` | `y_three` |
|------|----------|-----------|
| 静息 | 0 | 0 |
| 左手 | 1 | 1 |
| 右手 | 1 | 2 |

### 本期推荐做法（迁移学习，最容易落地）

```text
阶段 A：EEGNet(n_outputs=2)  + y_task  → 保存 best_task.pt
阶段 B：EEGNet(n_outputs=3)  + y_three → 加载 A 的「主干」权重
                                      → 只换掉最后一层分类器
                                      → 保存 best_three.pt
```

braindecode 的 `EEGNet` 里，`n_outputs=2` 与 `n_outputs=3` **只有最后一层形状不同**：

```text
相同（可迁移）: conv_* / bnorm_* 等主干
不同（必须重训）: final_layer.conv_classifier.weight / bias
                 2类: [2, 16, 1, 31] + bias[2]
                 3类: [3, 16, 1, 31] + bias[3]
```

### 文档已定、代码以后再接

- **被试独立五折 + 内层 val 早停**：协议见 `正式评估协议_被试独立五折.md`（先改文档，再改预处理/训练代码）
- 一个模型里**同时挂两个头**、一次前向出两套 logits（真正「共享主干双头并行」）
- 级联推理（先头1判静息/任务，任务才进头2）

---

## 1. 阶段 A：确认「第一头权重」怎么保存

你现在的 `train_task.py` **已经在保存**第一头权重。核心就是这段逻辑（对照你现有代码即可）：

**路径：** `code/train_lab/src/step/train_task.py`（已有，可微调文件名说明）

```python
# 验证集 F1 创新高时保存
best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
torch.save(
    {
        "stage": "A_braindecode_eegnet_task2",
        "backend": "braindecode.models.EEGNet",
        "n_outputs": 2,
        "model": best_state,          # ← 真正的权重字典
        "epoch": ep,
        "val_metrics": m,
    },
    OUT_DIR / "best_task.pt",         # ← 第一头检查点
)
```

### 你需要注意的 3 点

1. **先跑完阶段 A**，确认存在：  
   `code/train_lab/out/best_task.pt`
2. **阶段 B 不要覆盖这个文件**；第二头另存为 `best_three.pt`
3. 若你希望「每次都留一份带日期的备份」，可额外复制一份，例如：

```python
import shutil
from datetime import datetime

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(OUT_DIR / "best_task.pt", OUT_DIR / f"best_task_{stamp}.pt")
```

阶段 A 其它逻辑（`TaskHeadDataset`、`y_task`、`n_outputs=2`）**不用为了阶段 B 大改**。

---

## 2. 阶段 B 要新增/修改哪些文件

建议在现有结构上**增量**添加，不要把 `train_task.py` 揉成一团：

```text
code/train_lab/src/step/
├── dataset.py          ← 追加 ThreeHeadDataset（保留 TaskHeadDataset）
├── metrics.py          ← 追加三分类指标函数
├── train_task.py       ← 阶段 A（基本不动）
└── train_three.py      ← 【新建】阶段 B 入口
```

权重输出：

```text
code/train_lab/out/
├── best_task.pt        ← 阶段 A（保留）
└── best_three.pt       ← 阶段 B（新建）
```

数据仍读：

```text
code/preprocess_lab/out/bci2a/
├── train_X.npy / train_y_three.npy
└── val_X.npy   / val_y_three.npy
```

---

## 3. Dataset：读 `y_three`

**路径：** `code/train_lab/src/step/dataset.py`  
在文件末尾**追加**下面类（保留原来的 `TaskHeadDataset`）：

```python
class ThreeHeadDataset(Dataset):
    """
    阶段 B：空闲(0) / 左手(1) / 右手(2)。
    输出 x 形状 (8, 1000)，供 braindecode.EEGNet 使用。
    """

    def __init__(self, data_dir: str | Path, split: str = "train"):
        data_dir = Path(data_dir)
        X = np.load(data_dir / f"{split}_X.npy").astype(np.float32)
        # (N, 1, 8, 1000) → (N, 8, 1000)
        if X.ndim == 4 and X.shape[1] == 1:
            X = X[:, 0, :, :]
        assert X.ndim == 3 and X.shape[1:] == (8, 1000), X.shape
        self.X = X
        self.y_three = np.load(data_dir / f"{split}_y_three.npy").astype(np.int64)
        assert len(self.X) == len(self.y_three)
        # 可选自检：标签只能是 0/1/2
        assert set(np.unique(self.y_three)).issubset({0, 1, 2})

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (8, 1000)
        y = torch.tensor(self.y_three[idx], dtype=torch.long)
        return x, y
```

---

## 4. Metrics：三分类指标

**路径：** `code/train_lab/src/step/metrics.py`  
在文件末尾**追加**：

```python
def three_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """空闲=0 / 左手=1 / 右手=2。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # cm[i, j] = 真实 i、预测 j

    acc = accuracy_score(y_true, y_pred)
    # macro：三类平等平均，避免只被多数类抬高
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    recall_macro = recall_score(
        y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0
    )
    # 每类召回：看空闲/左/右各自漏了多少
    recall_per = recall_score(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "recall_macro": float(recall_macro),
        "recall_idle": float(recall_per[0]),
        "recall_left": float(recall_per[1]),
        "recall_right": float(recall_per[2]),
        "cm": cm,  # 打印用；保存 checkpoint 时可转 list
    }


def format_three_metrics(part_name: str, m: dict) -> str:
    cm = m["cm"]
    return "\n".join(
        [
            f"===== [{part_name}] 分类头2（空闲=0 / 左=1 / 右=2） =====",
            f"  混淆矩阵 (行=真实, 列=预测):",
            f"            pred0  pred1  pred2",
            f"    true0  {cm[0,0]:5d}  {cm[0,1]:5d}  {cm[0,2]:5d}",
            f"    true1  {cm[1,0]:5d}  {cm[1,1]:5d}  {cm[1,2]:5d}",
            f"    true2  {cm[2,0]:5d}  {cm[2,1]:5d}  {cm[2,2]:5d}",
            f"  Accuracy     = {m['accuracy']:.4f}",
            f"  F1-macro     = {m['f1_macro']:.4f}",
            f"  Recall-macro = {m['recall_macro']:.4f}",
            f"  Recall idle/left/right = "
            f"{m['recall_idle']:.4f} / {m['recall_left']:.4f} / {m['recall_right']:.4f}",
        ]
    )
```

阶段 B **默认按 `val` 的 `f1_macro` 选 best**（三类更均衡）。

---

## 5. 关键：从 `best_task.pt` 加载主干

这是阶段 B 与阶段 A 的「桥」。单独理解这段即可。

```python
def load_backbone_from_task_ckpt(
    model_three: nn.Module,
    ckpt_path: Path,
    freeze_backbone: bool = False,
) -> list[str]:
    """
    把阶段 A（n_outputs=2）的主干权重，装进阶段 B（n_outputs=3）模型。
    跳过 final_layer.conv_classifier.*（形状不同）。
    返回：成功加载的 key 列表，方便你打印核对。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt["model"]  # 与 train_task.py 保存时的字段一致
    dst = model_three.state_dict()

    loaded = []
    skipped = []
    new_state = {}
    for k, v in src.items():
        if k.startswith("final_layer.conv_classifier"):
            skipped.append(k)
            continue
        if k not in dst:
            skipped.append(k)
            continue
        if dst[k].shape != v.shape:
            skipped.append(k)
            continue
        new_state[k] = v
        loaded.append(k)

    # strict=False：允许 final_layer 仍是随机初始化
    missing, unexpected = model_three.load_state_dict(new_state, strict=False)
    # missing 里通常会看到 final_layer.conv_classifier.weight/bias —— 正常

    if freeze_backbone:
        for name, p in model_three.named_parameters():
            if not name.startswith("final_layer"):
                p.requires_grad = False

    print(f"loaded backbone tensors: {len(loaded)}")
    print(f"skipped: {skipped}")
    print(f"missing_keys (ok if only classifier): {missing}")
    print(f"unexpected_keys: {unexpected}")
    return loaded
```

### 要不要冻结主干？

| 策略 | 做法 | 适用 |
|------|------|------|
| **先不冻**（推荐入门） | 主干可微调 + 新分类头一起训 | 数据已有整库 BCI2A，通常更好 |
| 冻结主干 | `freeze_backbone=True`，只训 `final_layer` | 想验证「特征是否可迁移」、或数据很少时 |

建议：**先不冻跑一版**；若过拟合严重再试冻结。

---

## 6. 完整训练入口：`train_three.py`

**新建路径：** `code/train_lab/src/step/train_three.py`

> 风格刻意贴近你现有的 `train_task.py`，方便对照。

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from torch.utils.data import DataLoader

from dataset import ThreeHeadDataset
from metrics import format_three_metrics, three_class_metrics

# 本文件: MI/code/train_lab/src/step/train_three.py
# parents[0]=step → [1]=src → [2]=train_lab → [3]=code
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "preprocess_lab" / "out" / "bci2a"
OUT_DIR = ROOT / "train_lab" / "out"
TASK_CKPT = OUT_DIR / "best_task.pt"  # 阶段 A 权重（必须先存在）


def load_backbone_from_task_ckpt(
    model_three: nn.Module,
    ckpt_path: Path,
    freeze_backbone: bool = False,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt["model"]
    dst = model_three.state_dict()

    new_state = {}
    skipped = []
    for k, v in src.items():
        if k.startswith("final_layer.conv_classifier"):
            skipped.append(k)
            continue
        if k not in dst or dst[k].shape != v.shape:
            skipped.append(k)
            continue
        new_state[k] = v

    missing, unexpected = model_three.load_state_dict(new_state, strict=False)

    if freeze_backbone:
        for name, p in model_three.named_parameters():
            if not name.startswith("final_layer"):
                p.requires_grad = False

    print(f"[init] loaded {len(new_state)} tensors from {ckpt_path}")
    print(f"[init] skipped: {skipped}")
    print(f"[init] missing_keys: {missing}")
    print(f"[init] unexpected_keys: {unexpected}")


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))  # (B, 3)
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
            logits = model(x)
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
    print("TASK_CKPT:", TASK_CKPT)
    print("基线: braindecode.EEGNet | 阶段 B: n_outputs=3 | 加载头1主干")

    if not (DATA_DIR / "train_X.npy").exists():
        raise FileNotFoundError(f"找不到预处理数据: {DATA_DIR}")
    if not TASK_CKPT.exists():
        raise FileNotFoundError(
            f"找不到阶段 A 权重: {TASK_CKPT}\n请先运行 train_task.py 并确认已保存 best_task.pt"
        )

    train_loader = DataLoader(
        ThreeHeadDataset(DATA_DIR, "train"),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ThreeHeadDataset(DATA_DIR, "val"),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    # ★ 关键：三分类模型
    model = EEGNet(
        n_chans=8,
        n_outputs=3,
        n_times=1000,
        F1=8,
        D=2,
        F2=16,
        drop_prob=0.60,
    ).to(device)

    # ★ 关键：从第一头权重加载主干（先不冻）
    load_backbone_from_task_ckpt(model, TASK_CKPT, freeze_backbone=False)

    criterion = nn.CrossEntropyLoss()
    # 若 freeze_backbone=True，可改成只优化 requires_grad=True 的参数：
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), ...)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_score = -1.0
    best_state = None
    epochs = 100

    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        y_true, y_pred = collect_preds(model, val_loader, device)
        m = three_class_metrics(y_true, y_pred)

        print(f"\nep {ep:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(format_three_metrics("val", m))

        if m["f1_macro"] > best_score:
            best_score = m["f1_macro"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            # 保存时把 cm 转成 list，避免某些环境序列化问题
            m_save = {k: (v.tolist() if k == "cm" else v) for k, v in m.items()}
            torch.save(
                {
                    "stage": "B_braindecode_eegnet_three3",
                    "backend": "braindecode.models.EEGNet",
                    "n_outputs": 3,
                    "init_from": str(TASK_CKPT),
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": m_save,
                },
                OUT_DIR / "best_three.pt",  # ★ 不要覆盖 best_task.pt
            )
            print("  ↑ saved", OUT_DIR / "best_three.pt")

    if best_state is not None:
        model.load_state_dict(best_state)

    y_va, p_va = collect_preds(model, val_loader, device)
    print(format_three_metrics("val(best)", three_class_metrics(y_va, p_va)))
    y_tr, p_tr = collect_preds(model, train_loader, device)
    print(format_three_metrics("train(best)", three_class_metrics(y_tr, p_tr)))
    print("done. best val F1-macro =", best_score)


if __name__ == "__main__":
    main()
```

---

## 7. 怎么运行

在 `code/train_lab` 下（与阶段 A 相同环境）：

```powershell
# 1) 若还没有 best_task.pt，先跑阶段 A
python -m src.step.train_task

# 2) 确认权重存在
dir .\out\best_task.pt

# 3) 再跑阶段 B
python -m src.step.train_three
```

若你用 VS Code / Cursor 调试：可仿照现有 `launch.json` 再加一条：

```json
{
  "name": "train_three (stage B)",
  "type": "debugpy",
  "request": "launch",
  "module": "src.step.train_three",
  "cwd": "${workspaceFolder}/code/train_lab",
  "env": {
    "PYTHONPATH": "${workspaceFolder}/code/train_lab"
  }
}
```

（`.vscode` 被 gitignore 也没关系，本机自用即可。）

---

## 8. 验收清单（你自己改完后对照）

- [ ] `best_task.pt` 仍在，没有被阶段 B 覆盖  
- [ ] 新增 `ThreeHeadDataset`，读的是 `*_y_three.npy`  
- [ ] `EEGNet(..., n_outputs=3)`  
- [ ] 启动时打印 `loaded ... tensors`，且 `skipped` 含 `final_layer.conv_classifier.*`  
- [ ] 训练过程打印 3×3 混淆矩阵与 `F1-macro`  
- [ ] 产出 `out/best_three.pt`，里面 `n_outputs=3`，`init_from` 指向 `best_task.pt`  

快速自检加载是否成功（可选小脚本）：

```python
import torch
from pathlib import Path
from braindecode.models import EEGNet

ckpt_a = torch.load(Path("out/best_task.pt"), map_location="cpu")
ckpt_b = torch.load(Path("out/best_three.pt"), map_location="cpu")
print("A n_outputs:", ckpt_a["n_outputs"], "keys:", len(ckpt_a["model"]))
print("B n_outputs:", ckpt_b["n_outputs"], "init_from:", ckpt_b["init_from"])

m = EEGNet(n_chans=8, n_outputs=3, n_times=1000, F1=8, D=2, F2=16, drop_prob=0.6)
m.load_state_dict(ckpt_b["model"])
print("B load ok")
```

---

## 9. 常见坑

1. **`FileNotFoundError: best_task.pt`**  
   先跑通阶段 A；或 `DATA_DIR` / `OUT_DIR` 指错了盘符路径。

2. **把 `n_outputs=2` 的整包 `load_state_dict` 进 3 类模型**  
   会因 `final_layer` 形状不匹配报错。必须**跳过分类层**，或用本文的 `strict=False` 部分加载。

3. **阶段 B 仍读 `y_task`**  
   标签错了，三分类会训歪。Dataset 必须用 `y_three`。

4. **只看 Accuracy**  
   空闲/左/右也可能不均衡；优先看 **F1-macro** 和每类 Recall。

5. **误以为「保存了第一头」就要改模型结构成双头**  
   本期「保存第一头 + 跑第二头」= **两个检查点文件** + **迁移主干**；不是必须立刻合成一个双头网络。

---

## 10. 你改代码时的最小步骤（抄作业顺序）

1. 确认 `out/best_task.pt` 存在（没有就先跑 `train_task.py`）  
2. 在 `dataset.py` 追加 `ThreeHeadDataset`  
3. 在 `metrics.py` 追加 `three_class_metrics` / `format_three_metrics`  
4. 新建 `train_three.py`（整文件用第 6 节）  
5. 运行 `python -m src.step.train_three`  
6. 检查同时存在 `best_task.pt` 与 `best_three.pt`  

写完后若要我**只审阅、不直接改文件**：把三个文件内容或报错贴过来即可。

---

## 11. 以后可选：真正「一个模型两个头」长什么样（仅示意）

等迁移方案跑通，若你要贴近论文里的「共享主干双分类头」，结构示意如下（**本期不必实现**）：

```python
class DualHeadEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone = EEGNet 去掉 final_layer 的部分（需按 braindecode 结构拆）
        # self.head_task = ...   # → 2 类
        # self.head_three = ...  # → 3 类

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_task(feat), self.head_three(feat)
```

损失可以是：

```text
loss = loss_task + λ * loss_three
```

或分阶段：先只训 `head_task`，再训 `head_three` / 联合微调。  
那是下一阶段工程，**先把第 6 节的 `train_three.py` 跑通更重要**。
