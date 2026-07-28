# 技术方案 A：BCI2a + Stieger 简单拼接合并合训

> **【HISTORICAL】** 下文「历史对照」中若仍写迁主干，仅描述**已跑过夜**；新实验勿仿照。  
> 版本：v0.3（2026-07-27）  
> 状态：**代码已落地，合并过夜已跑完**（`overnight_20260727_141553`）  
> **注意**：该次过夜三分类仍为**旧策略（迁二分类主干）**。新约定「①独立训练 ②原生头 ③不迁权重」见 [`训练策略_二分类与三分类独立训练.md`](./训练策略_二分类与三分类独立训练.md)；**尚未改代码重跑**。  
> 对照：单库过夜见同目录 `20260727_上午单库过夜实验结果.md`；完整折表见 [`../模型训练/五折过夜实验记录_20260727_141553.md`](../模型训练/五折过夜实验记录_20260727_141553.md)  
> 原文件夹 `资料/实验_合并库合训方案A` 已删除，方案与结果统一归档本夹。

---

## 0. 一句话

把已对齐的 **BCI2a（2s/500）** 与 **Stieger（2s/500）** 离线 `concatenate` 成合并库，用现行 **被试独立五折 + 内层按人 val** 协议再跑一轮过夜；选型只看 Overall Val；报数同时给出 Overall / BCI2a-only / Stieger-only。

本方案 = 此前讨论的 **策略 A（简单拼接）**；不做按库采样、不做 loss 加权、不做域对抗（那些留作 B/C）。

---

## 1. 为什么 A 可行

两库现行预处理输出已统一：

| 项 | BCI2a | Stieger |
|----|-------|---------|
| 张量 | `(N,1,8,500)` | `(N,1,8,500)` |
| 通道序 | Cz…CPz 固定 8 导 | 同 |
| 标签 | `y_task` / `y_three` | 同约定 |
| 本机产物 | `out/bci2a_2s/bci2a_*.npy` | `out/stieger_2s/stieger_*.npy` |
| 规模（合并时） | N=1719，9 人 | N≈29k，13 人（随批次增长） |

合并后期望（以当前 `merge_meta.json` 为准）：

```text
N ≈ 31089
被试数 = 22（bci2a:9 + stieger:13）
形状 = (N, 1, 8, 500)
```

风险（接受并在报数里显式处理）：**训练会被 Stieger 主导**；因此必须做「按库分层五折 + 按库拆 test 指标」，否则 Overall 几乎等于 Stieger。

---

## 2. 离线合并产物（推荐先做脚本，不在过夜里现场拼）

### 2.1 输出目录

```text
code/preprocess_lab/out/merged_2s/
  merged_X.npy
  merged_y_task.npy
  merged_y_three.npy
  merged_subjects.npy
  merged_dataset.npy      # 可选："bci2a" | "stieger"
  merge_meta.json
```

### 2.2 合并规则

1. 分别加载两库全量 `*_X / *_y_task / *_y_three / *_subjects.npy`（`save_full=true` 产物）。
2. **被试 ID 加前缀**（强制）：
   - `A01` → `bci2a:A01`
   - `S1` → `stieger:S1`
3. 沿样本维 `np.concatenate`；可选同步写 `merged_dataset.npy`。
4. `merge_meta.json` 至少记录：来源路径、N、各类计数、被试列表、日期、方案版本 `A`。
5. **不覆盖** 原 `bci2a_2s/`、`stieger_2s/`。

建议落地文件（实现阶段再写，本阶段不建）：

```text
code/preprocess_lab/src/datasets/merge_bci2a_stieger.py
# 或 scripts/merge_datasets_2s.py
```

### 2.3 合并后自检清单

- [ ] `X.shape[1:] == (1, 8, 500)`，`dtype=float32`
- [ ] `len(X)==len(y_task)==len(y_three)==len(subjects)`
- [ ] `y_task∈{0,1}`，`y_three∈{0,1,2}`，且 `(y_three==0)==(y_task==0)`
- [ ] 唯一被试数 = 19；前缀无冲突
- [ ] `bci2a:*` 样本数 = 1719；`stieger:*` = 22195（以当时 out 为准）

---

## 3. 五折划分（方案 A 关键约束）

总协议不变（见 `正式评估协议_被试独立五折.md`）：

- 外层五折定 test（整人）
- 剩余人内层按人抽 ~20% 做 val
- 早停 / 选型 **只看 val**
- test **仅终评**

在合并库上增加 **按库分层组折**，避免某一折 test 全是 2a 或全是 Stieger：

```text
对 bci2a 的 9 人：固定 seed 均分成 5 组
对 stieger 的 10 人：固定 seed 均分成 5 组
第 k 折 test = 2a第k组 ∪ Stieger第k组
剩余人 → 再按人抽 val（建议仍按库分层抽，避免 val 也偏到单库）
```

同一人全部试次只落在 train / val / test 之一。

实现阶段：可在 `iter_subject_kfold` 旁新增 `iter_subject_kfold_stratified_by_dataset`，或合并脚本写出固定 `folds.json` 供训练读取。

---

## 4. 训练与过夜入口（实现时改动点）

仍复用现有流水线：

| 步骤 | 入口 |
|------|------|
| 头1 五折 | `code/train_lab/src/step/train_task_kfold.py` |
| 头2 五折 | `code/train_lab/src/step/train_three_kfold.py` |
| 过夜编排 | `code/train_lab/src/step/run_overnight_kfold.py` |

建议实现方式（仍属方案，未改代码）：

1. `DATA_DIR = .../out/merged_2s`，`DATA_PREFIX = "merged"`
2. 超参网格 **先沿用 Stieger 过夜同款**（样本量接近 Stieger）
3. `n_times` 从 `X.shape[-1]` 读取（500）
4. 环境：仓库根 `.venv`，device=cuda

### 4.1 报数扩展（方案 A 必做）

每个 fold 的 test 除 Overall 外，按 `subjects` 前缀拆：

| 块 | 含义 |
|----|------|
| Overall | 合并 test 上 Acc / F1（头1）或 F1-macro（头2） |
| BCI2a-only | 仅 `bci2a:*` |
| Stieger-only | 仅 `stieger:*` |

选型仍只用 **Overall Val**；按库 test 只报告、不参与选型。

### 4.2 对照实验（报数时并排）

固定同一 `seed=42`、同一网格逻辑时，三份对照：

1. 仅 BCI2a（今早 `20260727_102818`）
2. 仅 Stieger（今早 `20260727_110155`）
3. 合并库（本方案待跑）

解读重点：

- 合并后 **Stieger-only** 是否持平/略升
- **BCI2a-only** 是升（多数据帮到）还是降（被大库带偏）

---

## 5. 明确不做（本轮 A）

- 按库 batch 再平衡 / 过采样 2a
- 按库 loss 加权
- 域对抗 / 对齐损失
- 改在线采集窗长或 Phase4 协议
- 覆盖已有单库 `out/` 与已跑过夜权重

若合并后 BCI2a-only 明显变差，再开 **方案 B（再平衡）**，不在本文件范围。

---

## 6. 实施顺序（确认后改代码）

| 顺序 | 任务 | 产出 |
|------|------|------|
| 1 | 合并脚本 + 自检 | `out/merged_2s/` + `merge_meta.json` |
| 2 | 分层五折（或 folds.json） | 每折 test 含两库被试 |
| 3 | kfold/overnight 指向 merged；summary 按库拆分 | 代码改动 |
| 4 | 跑一夜 | `overnight_*` + `五折过夜实验记录_*_merged.md` |
| 5 | 写入本夹「合并过夜结果」小结 | 对照今早两轮 |

预计耗时：接近 Stieger 过夜量级（略增，因多约 7% 2a 样本）。

---

## 7. 验收标准（合并过夜跑完后）

- [x] 合并库形状与标签检查通过（`(31089,1,8,500)`，22 人）
- [x] 五折每折 test 同时含 `bci2a:*` 与 `stieger:*`（允许人数不均）
- [x] MD / summary 同时有 Overall 与按库指标
- [x] 选型依据仅为 Val；test 未参与网格选型
- [x] 与今早两轮单库结果可并排对照（见 §10）

---

## 8. 拍板记录

| 项 | 决定 |
|----|------|
| 策略 | **A：简单拼接** |
| 被试 ID | **`bci2a:` / `stieger:` 前缀** |
| 过夜范围 | 完整头1网格 + 头2（与现行 overnight 一致） |
| 代码 | **已实现**（见下「落地路径」） |

---

## 9. 落地路径（实现后）

| 组件 | 路径 |
|------|------|
| 合并脚本 | `code/preprocess_lab/src/datasets/merge_bci2a_stieger.py` |
| 合并产物 | `code/preprocess_lab/out/merged_2s/merged_*.npy` |
| 分层五折 | `iter_subject_kfold_stratified_by_dataset`（`split_subjects.py`） |
| 头1/头2 | `train_task_kfold.py` / `train_three_kfold.py` → `DATA_DIR=out/merged_2s` |
| 过夜 | `run_overnight_kfold.py` |

```text
cd D:\cyy\MI\code\preprocess_lab
$env:PYTHONPATH='.'
D:\cyy\MI\.venv\Scripts\python.exe -m src.datasets.merge_bci2a_stieger

cd D:\cyy\MI\code\train_lab\src\step
D:\cyy\MI\.venv\Scripts\python.exe run_overnight_kfold.py
```

---

## 10. 合并过夜结果小结（20260727_141553）

### 10.1 实验概况

| 项 | 内容 |
|----|------|
| 时间戳 | `20260727_141553` |
| 开始 / 结束 | `14:15:53` → `16:03:44`（约 1 小时 48 分） |
| device | `cuda` |
| 数据 | `out/merged_2s/merged_*.npy`，方案 A 简单拼接 |
| 规模 | N=31089，`(N,1,8,500)`，22 人（bci2a 9 + stieger 13） |
| 划分 | 按库分层五折；选型只看 Overall Val |
| 完整记录 | [`../模型训练/五折过夜实验记录_20260727_141553.md`](../模型训练/五折过夜实验记录_20260727_141553.md) |
| 权重根目录 | `code/train_lab/out/overnight_20260727_141553/` |

### 10.2 头1 网格（Val F1 选型）

| 网格 | 主要超参 | Val F1 | Test F1 Overall | Test F1 bci2a_only | Test F1 stieger_only |
|------|----------|--------|-----------------|--------------------|----------------------|
| G1 | lr=1e-3, drop=0.50, pat=15 | 0.8314 ± 0.0108 | 0.8248 ± 0.0208 | 0.7148 ± 0.1245 | 0.8285 ± 0.0197 |
| G2 | lr=7e-4, drop=0.55, pat=18 | **0.8340 ± 0.0099** | 0.8288 ± 0.0189 | 0.7691 ± 0.0445 | 0.8314 ± 0.0192 |
| G3 | lr=5e-4, drop=0.60, pat=20 | 0.8338 ± 0.0095 | 0.8301 ± 0.0172 | 0.7679 ± 0.0877 | 0.8319 ± 0.0181 |
| G4 | lr=1.5e-3, drop=0.40, pat=15 | 0.8332 ± 0.0113 | 0.8282 ± 0.0193 | 0.7832 ± 0.0422 | 0.8301 ± 0.0205 |

网格最优：**G2**。微调轮 Run02 的 Val 未超过 G2，故头1 最终取 **Run01 = G2**。

### 10.3 最终推荐（合并库）

**头1（静息/任务）· Run01**

| 指标 | mean±std |
|------|----------|
| Val F1 | **0.8340 ± 0.0099** |
| Test F1 Overall | 0.8288 ± 0.0189 |
| Test Acc Overall | 0.7116 ± 0.0254 |
| Test F1 bci2a_only | 0.7691 ± 0.0445 |
| Test F1 stieger_only | 0.8314 ± 0.0192 |

超参：`lr=7e-4`，`weight_decay=1e-4`，`drop_prob=0.55`，`patience=18`，`max_epochs=100`，`batch=32`，`seed=42`  
权重：`overnight_20260727_141553/00_task_grid_2/`

**头2（空闲/左/右）· Run04**（迁移最优头1）

| 指标 | mean±std |
|------|----------|
| Val F1-macro | **0.4371 ± 0.0320** |
| Test F1-macro Overall | 0.4109 ± 0.0409 |
| Test Acc Overall | 0.4224 ± 0.0396 |
| Test F1-macro bci2a_only | 0.4304 ± 0.0496 |
| Test F1-macro stieger_only | 0.4067 ± 0.0462 |

超参：`lr=1.5e-3`，`weight_decay=1e-4`，`drop_prob=0.5`，`patience=20`，`freeze_backbone=False`  
权重：`overnight_20260727_141553/04_three_tuned/`

### 10.4 与今早单库过夜对照

| 项目 | 仅 BCI2a `102818` | 仅 Stieger `110155` | **合并 A `141553`** |
|------|-------------------|---------------------|---------------------|
| N / 人数 | 1719 / 9 | 22195 / 10（当时） | 31089 / 22 |
| 头1 Val F1 | 0.8262 ± 0.0160 | 0.8443 ± 0.0159 | 0.8340 ± 0.0099 |
| 头1 Test F1 Overall | 0.7777 ± 0.0489 | 0.8301 ± 0.0202 | **0.8288 ± 0.0189** |
| 头1 Test F1（该库子集） | — | — | bci2a **0.7691** / stieger **0.8314** |
| 头2 Val F1-macro | 0.5323 ± 0.0674 | 0.4357 ± 0.0375 | 0.4371 ± 0.0320 |
| 头2 Test F1-macro Overall | 0.4454 ± 0.0985 | 0.3992 ± 0.0076 | **0.4109 ± 0.0409** |
| 头2 Test F1-macro（该库子集） | — | — | bci2a **0.4304** / stieger **0.4067** |

### 10.5 解读（方案 A）

1. **Overall 接近 Stieger 单库**：头1 Test F1 ≈ 0.83，与 Stieger 过夜几乎持平；说明合并后主信号仍由大库主导（符合方案 A 预期）。
2. **BCI2a-only 头1**：合并后 Test F1 ≈ 0.77，与单库 BCI2a 过夜 Test F1（0.78）接近，**未见明显被大库带崩**；但也没有相对单库明显提升。
3. **Stieger-only 头1**：合并后 ≈ 0.83，与单库 Stieger 持平，合训未伤大库头1。
4. **头2 仍难**：Overall F1-macro ≈ 0.41，介于两单库之间；bci2a_only 子集（0.43）略低于该库单库过夜点估计（0.45），折间方差仍大。
5. **下一步**：若目标是抬高 BCI2a-only，再开方案 B（按库再平衡 / 加权）；本轮 A 作为池化合训基线可冻结。
