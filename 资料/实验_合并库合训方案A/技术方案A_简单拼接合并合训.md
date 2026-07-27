# 技术方案 A：BCI2a + Stieger 简单拼接合并合训

> 版本：v0.2（2026-07-27）  
> 状态：**代码已落地**（合并脚本 + 分层五折 + 训练读 merged）  
> 对照：今早已完成单库过夜（见同目录 `20260727_上午单库过夜实验结果.md`）

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

- [ ] 合并库形状与标签检查通过
- [ ] 五折每折 test 同时含 `bci2a:*` 与 `stieger:*`（允许人数不均）
- [ ] MD / summary 同时有 Overall 与按库指标
- [ ] 选型依据仅为 Val；test 未参与网格选型
- [ ] 与今早两轮单库结果可并排对照

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
