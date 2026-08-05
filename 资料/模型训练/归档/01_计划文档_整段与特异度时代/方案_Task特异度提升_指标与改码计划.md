> **【已归档】** 非现行主线。请改读 资料/模型训练/00_当前主线_2s滑窗100ms/实验操作手册.md（旧1s见归档/04）。

# Task 基线：静息 / 任务能否分清——指标刻度 + 精简改码计划

> 状态：**仅文档，尚未改训练代码**（按本文确认后再动仓库）。  
> 完整可对照实例代码见 **§9**（只写在本 MD，**未**创建/修改项目内任何 `.py`）。  
> 数据设定：`merged_2s`，被试独立五折，`seed=42`（与 `shared_hparams.py` 一致）。  
> **目标定位（按当前需求）**：各模型只作**基线对照**；二分类 Task **只用来回答**——静息态与任务态能否识别清楚。  
> **不在本次范围**：在线部署、时间平滑 C2、锁主模型写部署说明、大消融表、Three 专项优化、SMOTE / Focal / 八模型齐改。

---

## 0. 需求与范围

| 要回答的问题 | 判据 | 不回答的问题 |
|--------------|------|--------------|
| 基线模型能否把**静息 vs 任务**分清？ | Spec + Recall + BalAcc（见 §2、§3.0） | 谁 F1 最高、能否上线演示、左右手解码好不好 |

**最少工作流（本次只需这些）**

1. 定「分得清」的数值标准  
2. 改训练目标：加权 CE + BalAcc 早停（仅 Shallow + EEGTCNet）  
3. 跑五折，对照旧基线，看过不过关；不过则只调 `w0`  
4. （可选）在 val 上扫阈值 τ，看分清程度还有没有余量  

Three 仍可随脚本顺带跑出数字，但**不作为本计划验收对象**；改码时 **Three 分支不动**。

---

## 1. 假象门槛（读表前必看）

| 现象 | 数值含义 |
|------|----------|
| 任务占比约 **71%**、静息约 **29%** | 全判「任务」→ Acc≈**0.71**、F1≈**0.83**、Spec≈**0**、BalAcc≈**0.50** |
| 全判「静息」 | Acc≈**0.29**、Rec≈**0**、Spec≈**1**、BalAcc≈**0.50** |

**结论**：只看 Acc≈0.71 或 F1≈0.83 **不能**说「分得清」；必须同时看 Spec / Rec / BalAcc。  
旧基线 F1 好看但 Spec≈0 → **尚未分清静息**，只是偏爱判任务。

---

## 2. 指标刻度表（本项目用）

口径：五折 **均值**；单折可波动更大。

### 2.1 Task（静息 vs 任务）——本次主表：「能否分清」

| 指标 | 差 / 没分清 | 旧基线常见（虚高） | **分得清（本次过关）** | 分得很清（更好） |
|------|-------------|--------------------|------------------------|------------------|
| **Specificity**（静息） | &lt;0.15 | 0.15–0.35 | **≥0.40** | ≥0.60 |
| **Recall**（任务） | &lt;0.70 | 常 &gt;0.90（单边） | **≥0.75** | ≥0.80 且 Spec 仍高 |
| **Balanced Acc** | ≈0.50 | 0.55–0.64 | **≥0.65** | ≥0.70 |
| **Test F1** | &lt;0.70 或乱飘 | 0.78–0.84（虚高区） | **允许降到 ~0.72–0.80** | 附报即可，不作主结论 |
| **Test Acc** | 随机或单边 | ≈0.68–0.72（易虚高） | 附报；须 Spec/Rec 双过 | 附报 |

**旧基线粗画像（改目标前）**

- F1≈0.82–0.83、Acc≈0.71 → 表面「优」  
- Spec 多在 0.00–0.15、BalAcc 常 ≈0.50–0.55 → **没分清静息**  
- 图 bandpower：F1 仍可 ≈0.83，Spec≈0 → 不能当「已分清」的证据  

**本次一票否决（写进基线结论时）**

- Spec 五折均值 &lt;0.20，或多数折 TN≈0 → 结论写「未分清」  
- 为抬 Spec 导致 Rec &lt;0.70 → 也算未分清（塌成全静息）  
- 只用 F1 排名宣称「基线已能区分静息/任务」→ 不允许  

**结论写法（基线报告）**

1. 先看是否过关：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65  
2. 过关 →「在加权 CE + BalAcc 选模下，该基线可初步分清静息/任务」  
3. 不过 →「当前设定下仍未分清；F1 高不能替代 Spec」；再只调 `w0` 或查数据窗（§3.3）

### 2.2 Three（idle / left / right）——仅参考，本次不验收

随机三分类 Acc≈0.33；现基线宏 F1 约 0.36–0.42，属**弱基线带、结果形态正常**，但与「静息/任务能否分清」无关。

| 指标 | 差 | 当前基线带 | 可接受（研究，非本次） |
|------|-----|------------|------------------------|
| **F1-macro** | &lt;0.35 | **0.36–0.42** | ≥0.45 |
| **Acc** | ≈0.33–0.37 | **0.37–0.44** | ≥0.45 |
| **Idle Recall** | &lt;0.25 | 常偏低 | ≥0.40 |

说明：Shallow≈0.42 F1m 是基线里相对最好，不是本次目标。Three 专项以后再说。

### 2.3 辅助观察（不当唯一分数）

| 观察 | 差的信号 | 正常 / 可接受 |
|------|----------|----------------|
| `bci2a_only` vs `stieger_only` | 差 &gt;0.10 且总是 BCI2a 崩 | Stieger 略高 0.02–0.08（常见） |
| Task 最优轮次 | 极早「最优」+ Spec≈0 | 数轮～数十轮，且 Spec/BalAcc 有改善 |

---

## 3. 精简执行计划（按当前需求）

### 3.0 成功标准（写进基线协议）

| 指标 | 本次过关（分得清） |
|------|--------------------|
| Spec（五折均值） | ≥ **0.40** |
| Recall（任务） | ≥ **0.75**（底线 0.70） |
| Balanced Acc | ≥ **0.65** |
| Test F1 | 允许从 ~0.83 → ~0.75；**只附报** |

强制：

- Task 早停 / 是否「分得清」：**禁止只看 F1**；盯 **Balanced Acc** + Spec + Rec  
- 每张 Task 表同时报：Spec / Rec / BalAcc / F1 / Acc  
- 实验名带后缀：`_wce2_balacc`，禁止与旧 F1 早停 run 混表  

### 3.1 步骤 A（必做）：改训练目标，回答「能否分清」

对应原方案 R1 / Phase-1，**这是本次唯一必做实验臂**。

**做**

1. **加权 CE（A1）**：静息 `w0=2.0`，任务 `w1=1.0`（起步）  
2. **早停改 BalAcc（A3a）**：`best_score = m["balanced_accuracy"]`  
3. **只改两条基线脚本**：`shallow` + `eegtcnet`（同一 `merged_2s`、同一 seed）  
4. **Three 分支不改**（脚本可仍跑 Three，但不改损失/早停，不作验收）

**不做**

- SMOTE、Focal、平衡采样、换结构、图模型重训、八模型齐改  
- `w0≥4` 的极端加权  
- 部署、C2 平滑、锁主模型  

**验收**

- 两模型任一或两者达到 §3.0 → 可写「基线在修正目标后能初步分清」  
- Spec 仍 &lt;0.35 → **只调** `w0∈{2.5, 3.0}` 再跑，不同时加别的手段  
- Spec≥0.5 但 Rec&lt;0.70 → 降到 `w0=1.5`  
- 调权后仍长期 Spec 很低 → 转 §3.3 查数据可分性，而不是先换模型  

**为何只改两条**：验证的是「目标/选模」是否让静息被认出，不是刷全模型榜；Shallow=稳基线，EEGTCNet=旧 F1 虚高代表。

### 3.2 步骤 B（可选）：阈值扫描，看「分清余量」

- **不重训**：加载步骤 A 的 `best_task.pt`  
- 在各折 **val** 上扫 \(\tau\)：\(P(\text{任务})\ge\tau\) 才判任务  
- 用途：看 Spec–Rec 曲线，辅助描述「分得清到什么程度」  
- **不做**：C2 时间平滑、写部署说明、为上线定唯一 τ（除非以后有演示需求再单开）

### 3.3 步骤 C（仅当 A 反复不过关）：查数据窗

- 抽被试对比静息窗 vs 任务窗（谱 / 时域）  
- 若本身糊：收紧静息定义或换窗；继续分报 `bci2a_only` / `stieger_only`  
- **不挡**步骤 A 先做；A 过关则可不做 C  

### 3.4 本次明确不做（原 Phase-2/3 工程与研究项）

| 原内容 | 本次态度 |
|--------|----------|
| C2 连续 K 窗平滑 | 不做（演示/在线才需要） |
| 锁 Shallow + 部署说明 | 不做（非上线目标） |
| Focal / B1 / SMOTE 消融 | 不做 |
| 图模型 + Encoder 对照臂 | 不做 |
| Three idle 加权专项 | 不做 |
| 八个 baseline 全部改加权 | 不做；A 过关后若要统一基线协议，再另开文档批量补丁 |

### 3.5 为何不用 SMOTE（备忘）

- 问题是「目标偏任务 + F1 选模」，不是静息少到必须造样本  
- raw EEG 上 SMOTE 难定义近邻、难复现  
- 与「基线是否分得清」无关的复杂度，本次不加  

---

## 4. 代码修改落点（确认后再改；本文不落地）

路径均相对仓库根 `MI/`。

### 4.1 必改 / 新建（步骤 A）

| 优先级 | 文件 | 现状（约） | 计划改动 |
|--------|------|------------|----------|
| **新建** | `code/train_lab/src/step/baselines_single/task_objective.py` | 不存在 | `build_task_ce(w0=2, w1=1)` 或按 train 折计数；返回加权 `CrossEntropyLoss` |
| **必改** | `code/train_lab/src/step/baselines_single/baseline_shallow.py` | Task：`CrossEntropyLoss()`（约 L184）；早停 `m["f1"]`（约 L195–196）；MD/命名（约 L410+） | Task 用加权 CE；早停 BalAcc；日志 Spec/Rec/BalAcc/F1；后缀 `_wce2_balacc` |
| **必改** | `code/train_lab/src/step/baselines_single/baseline_eegtcnet.py` | 与 shallow 同构 | **相同改法** |
| **建议改** | `code/train_lab/src/step/baselines_single/md_fold_detail.py` | Val 仍写「Val F1（最优）」 | 改为「Val 选模=BalAcc」；展示最优 BalAcc 及当时 F1 |
| **可选** | `shared_hparams.py` / `run_all_baseline_model.py` | 共用 HP / 串跑 | 常量 `w0` 或 `--models shallow,eegtcnet` 即可 |

**一般不用改**：`metrics.py`（已有 Spec/BalAcc）、`dataset.py`、划分与 `data_paths.py`。

### 4.2 明确不动

| 文件 / 分支 | 原因 |
|-------------|------|
| `baseline_eegnet/deep/conformer/dbn/gcbnet/dgcnn.py` | 本次只验证两条基线 |
| `Self_development_model/TepmoralEncoder_*.py` | 非基线「分清」最小集 |
| `train_three_one_fold`（CE / F1-macro 早停） | 与本次问题无关 |

### 4.3 可选新建（步骤 B）

| 文件 | 计划 |
|------|------|
| `code/train_lab/src/step/baselines_single/eval_threshold_sweep.py` | 加载各折 `best_task.pt`，val 扫 τ，出 Spec/Rec 曲线；test 只终报 |

### 4.4 命名约定

```text
out/baseline/shallow_wce2_balacc/merged_2s/run_<stamp>/
资料/模型训练/runs/<stamp>_shallow_wce2_balacc/
```

### 4.5 实施勾选（按需求精简）

1. [ ] 新建 `task_objective.py`（实例见 §9.1）  
2. [ ] 改 `baseline_shallow.py` Task（实例见 §9.2）  
3. [ ] 同步改 `baseline_eegtcnet.py`（同 §9.2，仅 `MODEL_NAME` / stage 字符串不同）  
4. [ ] 微调 `md_fold_detail.py`（实例见 §9.3）  
5. [ ] 跑 shallow 五折 → 对照旧 Spec/Rec/BalAcc  
6. [ ] 跑 eegtcnet 五折 → 写「是否分得清」结论  
7. [ ] 若不过关：只调 `w0` 再跑  
8. [ ] （可选）`eval_threshold_sweep.py`（实例见 §9.4）  
9. [ ] （仅 A 多次失败）静息窗抽检  

---

---

## 5. 实验清单（对齐需求）

**必做**

- [ ] 确认 §3.0「分得清」标准  
- [ ] 落地步骤 A（shallow + eegtcnet，`w0=2` + BalAcc）  
- [ ] 对比表：旧 F1 早停基线 vs 新 run（主列 Spec/Rec/BalAcc）  
- [ ] 用一句话写结论：分得清 / 未分清  

**可选**

- [ ] val 阈值扫描（步骤 B）  
- [ ] 调 `w0` 一轮  
- [ ] 数据窗抽检（步骤 C）  

**不做（除非需求变更）**

- [ ] ~~C2 / 部署说明~~  
- [ ] ~~Focal / SMOTE / 全模型消融~~  
- [ ] ~~Three idle 专项~~  

---

## 6. 失败模式（基线结论防踩坑）

| 做法 | 问题 |
|------|------|
| 继续用 F1 宣称「已能区分静息/任务」 | 虚高，Spec 可能仍≈0 |
| 只加压极大 `w0` | 易全判静息，Rec 崩 |
| 没改目标就换模型 | 基线 Spec 通病仍在 |
| 把 Three 分数和 Task「分得清」混谈 | 两任务独立，问题不同 |

---

## 7. 默认拍板（待确认后改代码）

- **目标**：基线层面回答「静息 vs 任务能否分清」  
- **必做**：`w0=2` 加权 CE + BalAcc 早停，仅 **Shallow + EEGTCNet**，`merged_2s` / seed=42  
- **可选**：val 阈值扫描  
- **不做**：C2、部署、大消融、Three 专项、SMOTE  
- **超参**：沿用 `shared_hparams`（`lr=1e-4`, `drop_prob=0.5` 等）  

确认后按 §4.5 / §9 改仓库；**完整实例仅写在本文 §9，尚未写入项目任何 `.py` 文件。**

---

## 8. 相关现有文档 / 记录（只读参考）

- 汇总报告：`资料/模型训练/runs/汇总/.../多模型被试独立五折交叉验证实验汇总报告*.md`  
- 逐折明细：`资料/模型训练/runs/20260730_232446_shallow/`、`.../20260731_105045_eegtcnet/` 等  
- 最新入口：`资料/模型训练/五折实验记录_最新.md`  

---

## 9. 完整实例代码（仅文档；勿直接当已落地代码）

> **说明**  
> - 以下供审阅与对照；**当前仓库中尚未创建/修改这些文件**。  
> - 对齐现有 `baseline_shallow.py` / `baseline_eegtcnet.py` 结构与 `metrics.binary_task_metrics` 字段。  
> - 标签约定：静息=`0`，任务=`1`。  
> - `eegtcnet` 与 `shallow` 改法相同，仅替换 `MODEL_NAME`、checkpoint 里 `stage` 字符串（如 `task2_eegtcnet`）。

### 9.1 新建 `task_objective.py`（完整实例）

目标路径：`code/train_lab/src/step/baselines_single/task_objective.py`

```python
"""Task 二分类训练目标：加权交叉熵（文档实例，落地前复制到 baselines_single/）。

默认固定 w0=2, w1=1（加重静息）。也可按 train 折标签做逆频率加权。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fixed_task_weights(w0: float = 2.0, w1: float = 1.0) -> torch.Tensor:
    """返回 shape (2,)：index0=静息, index1=任务。"""
    return torch.tensor([float(w0), float(w1)], dtype=torch.float32)


def inverse_freq_task_weights(y_train: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    """逆频率：w_c = N / (C * n_c)，再可选归一化到均值 1。"""
    y = np.asarray(y_train).astype(int).reshape(-1)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n_total = float(counts.sum())
    w = n_total / (n_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def build_task_ce(
    device: torch.device,
    *,
    mode: str = "fixed",
    w0: float = 2.0,
    w1: float = 1.0,
    y_train: np.ndarray | None = None,
) -> nn.CrossEntropyLoss:
    """
    mode:
      - "fixed": 使用 w0/w1
      - "invfreq": 使用 y_train 逆频率（需传入该折 train 标签）
    """
    if mode == "fixed":
        weight = fixed_task_weights(w0=w0, w1=w1)
    elif mode == "invfreq":
        if y_train is None:
            raise ValueError("mode='invfreq' 需要 y_train")
        weight = inverse_freq_task_weights(y_train)
    else:
        raise ValueError(f"未知 mode={mode}")
    return nn.CrossEntropyLoss(weight=weight.to(device))
```

### 9.2 改 `train_task_one_fold`（Shallow 完整替换实例）

在 `baseline_shallow.py` 顶部增加：

```python
from task_objective import build_task_ce

# 与旧 run 区分；也可只在 out 目录名加后缀而不改 MODEL_NAME
MODEL_NAME = "shallow_wce2_balacc"
TASK_W0 = 2.0
TASK_W1 = 1.0
TASK_WEIGHT_MODE = "fixed"  # 或 "invfreq"
```

将现有 `train_task_one_fold` **整体替换为**（Three 函数保持不动）：

```python
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
    model = build_model(8, int(X.shape[-1]), 2, hp.drop_prob).to(device)

    # ---- 相对旧代码：加权 CE（加重静息）----
    y_tr = y[masks["train"]]
    criterion = build_task_ce(
        device,
        mode=TASK_WEIGHT_MODE,
        w0=TASK_W0,
        w1=TASK_W1,
        y_train=y_tr,
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

        # ---- 相对旧代码：日志多报 Spec/Rec/BalAcc；早停盯 BalAcc ----
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
                    "stage": "task2_shallow",
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "n_outputs": 2,
                    "weight_transfer": False,
                    "classifier": "native",
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": jsonify_metrics(m),
                    "hparams": {
                        **shared_as_dict(),
                        "task_weight_mode": TASK_WEIGHT_MODE,
                        "task_w0": TASK_W0,
                        "task_w1": TASK_W1,
                        "task_early_stop": "balanced_accuracy",
                    },
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
        "best_val_f1": float(best_val_f1),  # 对照用；选模不是 F1
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "test_metrics": m_te,
        "test_metrics_by_dataset": by_ds,
    }
```

`run_task_kfold` 汇总处建议同时打印 BalAcc（实例片段）：

```python
def run_task_kfold(X, y, subjects, device, hp: SharedTrainHP, out_dir: Path, data_tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for info in iter_folds(subjects, hp, data_tag):
        folds.append(train_task_one_fold(info, X, y, subjects, device, hp, out_dir))

    val_bal = [r["best_val_balanced_accuracy"] for r in folds]
    test_spec = [r["test_metrics"]["specificity"] for r in folds]
    test_rec = [r["test_metrics"]["recall"] for r in folds]
    test_bal = [r["test_metrics"]["balanced_accuracy"] for r in folds]
    test_f1s = [r["test_metrics"]["f1"] for r in folds]

    def ms(xs):
        return _mean_std(xs)

    print(
        f"\n[TASK] Val BalAcc {ms(val_bal)[0]:.4f}±{ms(val_bal)[1]:.4f} | "
        f"Test Spec {ms(test_spec)[0]:.4f}±{ms(test_spec)[1]:.4f} | "
        f"Test Rec {ms(test_rec)[0]:.4f}±{ms(test_rec)[1]:.4f} | "
        f"Test BalAcc {ms(test_bal)[0]:.4f}±{ms(test_bal)[1]:.4f} | "
        f"Test F1 {ms(test_f1s)[0]:.4f}±{ms(test_f1s)[1]:.4f}"
    )

    # …其余 summary 字段按项目习惯写入；务必包含 test_spec/rec/balacc 均值±std
    return {
        "model_name": MODEL_NAME,
        "folds": folds,
        "val_balanced_accuracy_mean": ms(val_bal)[0],
        "val_balanced_accuracy_std": ms(val_bal)[1],
        "test_specificity_mean": ms(test_spec)[0],
        "test_specificity_std": ms(test_spec)[1],
        "test_recall_mean": ms(test_rec)[0],
        "test_recall_std": ms(test_rec)[1],
        "test_balanced_accuracy_mean": ms(test_bal)[0],
        "test_balanced_accuracy_std": ms(test_bal)[1],
        "test_f1_mean": ms(test_f1s)[0],
        "test_f1_std": ms(test_f1s)[1],
    }
```

`main()` 里 MD 抬头建议写明选模指标（片段）：

```python
# 原：f"- model：`{MODEL_NAME}`（单脚本；无 registry）"
# 改为增加一行：
f"- Task 目标：加权CE w0={TASK_W0}, w1={TASK_W1}, mode={TASK_WEIGHT_MODE}；早停=Balanced Acc",
f"- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）",
```

**EEGTCNet**：复制同一套改动到 `baseline_eegtcnet.py`，设 `MODEL_NAME = "eegtcnet_wce2_balacc"`，checkpoint `"stage": "task2_eegtcnet"`。

### 9.3 `md_fold_detail.py` 文案实例（Task 折明细）

将 `_task_fold_lines` 中与「Val F1（最优）」相关的几行改为：

```python
def _task_fold_lines(r: dict) -> list[str]:
    m = r.get("test_metrics") or {}
    fold = r.get("fold", "?")
    lines = [
        f"#### Fold {fold}",
        "",
        f"- 早停/结束轮次（stopped_epoch）：`{r.get('stopped_epoch')}`",
        f"- 验证最优轮次（best_epoch）：`{r.get('best_epoch')}`",
        f"- Val 选模分数（Balanced Acc）：`{_f(r.get('best_val_balanced_accuracy'))}`",
        f"- Val F1（最优 checkpoint 时，附报）：`{_f(r.get('best_val_f1'))}`",
        f"- Val loss（最优时）：`{_f(r.get('best_val_loss'))}`",
        "",
        "**Test（overall）**",
        f"- Accuracy：`{_f(m.get('accuracy'))}`",
        f"- Recall：`{_f(m.get('recall'))}`",
        f"- Specificity：`{_f(m.get('specificity'))}`",
        f"- Precision：`{_f(m.get('precision'))}`",
        f"- F1：`{_f(m.get('f1'))}`",
        f"- Balanced Acc：`{_f(m.get('balanced_accuracy'))}`",
        f"- 混淆矩阵：TP=`{m.get('tp')}` TN=`{m.get('tn')}` FP=`{m.get('fp')}` FN=`{m.get('fn')}`",
        "",
    ]
    # … by_dataset 段落保持原逻辑
    return lines
```

### 9.4 可选：`eval_threshold_sweep.py`（完整实例）

目标路径：`code/train_lab/src/step/baselines_single/eval_threshold_sweep.py`  
用途：步骤 B；**不训练**。需能重建与训练时相同的 val/test loader 与 `build_model`（此处以 Shallow 为例；TCNet 换 import 即可）。

```python
"""对已保存的 best_task.pt 做 P(任务)≥τ 阈值扫描（文档实例，未写入仓库）。

用法示意（落地后）：
  python eval_threshold_sweep.py --run_dir <.../run_xxx> --data merged_2s --model shallow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 落地时按项目 path 习惯补 sys.path；此处省略
from metrics import binary_task_metrics
from dataset import ArrayTaskDataset


@torch.no_grad()
def collect_proba(model: nn.Module, loader: DataLoader, device: torch.device):
    """返回 y_true (N,), p_task (N,) = softmax 的任务类概率。"""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics_at_tau(y_true: np.ndarray, p_task: np.ndarray, tau: float) -> dict:
    y_pred = (p_task >= tau).astype(int)
    return binary_task_metrics(y_true, y_pred)


def sweep_taus(y_true, p_task, taus=None) -> list[dict]:
    if taus is None:
        taus = [round(x, 2) for x in np.arange(0.50, 0.86, 0.05)]
    rows = []
    for tau in taus:
        m = metrics_at_tau(y_true, p_task, tau)
        rows.append(
            {
                "tau": float(tau),
                "specificity": m["specificity"],
                "recall": m["recall"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1": m["f1"],
                "accuracy": m["accuracy"],
            }
        )
    return rows


def pick_tau_on_val(rows: list[dict], *, min_recall: float = 0.75) -> dict | None:
    """在 Rec≥阈值的候选里选 BalAcc 最高；若无则放宽为 BalAcc 最高。"""
    ok = [r for r in rows if r["recall"] >= min_recall]
    pool = ok if ok else rows
    return max(pool, key=lambda r: r["balanced_accuracy"]) if pool else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True, help="含 task/fold*/best_task.pt 的 run 目录")
    p.add_argument("--min_recall", type=float, default=0.75)
    args = p.parse_args()

    # ---- 落地时在此：load X,y,subjects；按与训练相同的 fold 划分建 val/test loader；build_model+load state ----
    # 伪流程（请接到现有 resolve_data / iter_folds / build_model）：
    #
    # for fold in 0..4:
    #   ckpt = torch.load(run_dir / "task" / f"fold{fold}" / "best_task.pt", map_location=device)
    #   model.load_state_dict(ckpt["model"])
    #   y_val, p_val = collect_proba(model, val_loader, device)
    #   rows_val = sweep_taus(y_val, p_val)
    #   best = pick_tau_on_val(rows_val, min_recall=args.min_recall)
    #   y_te, p_te = collect_proba(model, test_loader, device)
    #   m_te = metrics_at_tau(y_te, p_te, best["tau"])
    #   打印 / 写入 JSON：val 曲线 + test 终报
    #
    # 过关仍以「训练时默认 τ=0.5（argmax）的 test Spec/Rec/BalAcc」为主；
    # 本脚本仅辅助看余量。

    print("文档实例：请按注释接到现有数据与模型构建后再运行。")
    print(f"run_dir={args.run_dir} min_recall={args.min_recall}")


if __name__ == "__main__":
    main()
```

**单折手动验算示例**（理解 C1，不必落盘）：

```python
# 假设 val 上已有 y_true, p_task
for tau in (0.5, 0.6, 0.7, 0.8):
    y_pred = (p_task >= tau).astype(int)
    m = binary_task_metrics(y_true, y_pred)
    print(tau, m["specificity"], m["recall"], m["balanced_accuracy"])
```

### 9.5 与现状差异速查

| 位置 | 旧代码 | 实例代码 |
|------|--------|----------|
| `criterion` | `nn.CrossEntropyLoss()` | `build_task_ce(..., w0=2)` |
| 早停 | `m["f1"]` | `m["balanced_accuracy"]` |
| 日志 | 只打 `val_F1` | BalAcc / Spec / Rec / F1 |
| 返回 | `best_val_f1` 实为选模分 | `best_val_balanced_accuracy` + 附报 F1 |
| `MODEL_NAME` / 目录 | `shallow` | `shallow_wce2_balacc`（防混表） |
| Three | 同脚本另训 | **不改** |
| 阈值脚本 | 无 | §9.4 可选 |

### 9.6 跑通后怎么写结论（模板）

```text
数据：merged_2s，被试独立五折，seed=42
设置：Task 加权CE w0=2,w1=1；早停=Balanced Acc；模型=shallow_wce2_balacc / eegtcnet_wce2_balacc
结果：Test Spec=…±…，Rec=…±…，BalAcc=…±…，F1=…±…（附报）
结论：达到 / 未达到「Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65」→ 静息与任务【能 / 不能】初步分清。
```
