# Phase 4：采一轮 → 切窗 → 训练检查清单

> **版本**：v0.1  
> **日期**：2026-07-23  
> **配套**：`marker_spec.md`、`offline/`、`preprocess_lab`、`train_lab`

---

## 1. 目标

把 `experiment_game` 会话（`eeg.csv` + `events.jsonl`）切成与 `preprocess_lab` 一致的：

| 数组 | 形状 | 含义 |
|------|------|------|
| `X` | `(N, 1, 8, 1000)` | 4 s × 250 Hz，8 通道 |
| `y_task` | `(N,)` | 0=Rest，1=Task(左或右) |
| `y_three` | `(N,)` | 0=Rest，1=Left，2=Right |

切窗锚点：**`mi_start` / `rest_start`**（不用 `cue`，避免把 Cue 展示段切进训练窗）。  
默认只保留 **`phase == acquire`**，并丢弃 `trial_reject` 的 trial。

---

## 2. 通道对齐

| 采集（eeg.csv） | preprocess 目标 |
|-----------------|-----------------|
| CZ | Cz |
| C3 | C3 |
| C4 | C4 |
| CP3 / CPZ3 | CP3 |
| CP4 / CPZ4 | CP4 |
| CPZ | CPz |
| FC3 | FC3 |
| FC4 | FC4 |

输出列顺序固定：`C3, C4, Cz, CP3, CP4, CPz, FC3, FC4`。

---

## 3. 一键切窗

在仓库根、使用 lsl_connect 的 venv（需已 `pip install scipy`）：

```powershell
cd d:\cyy\MI

.\collect_data\LSL_connect_model\LSL_connect_model\.venv\Scripts\python.exe `
  -m experiment_game.tools.run_phase4_epochs `
  --session experiment_game\data\sessions\<会话目录名>
```

输出目录默认：

```text
experiment_game/data/epochs/<会话目录名>/
  X.npy
  y_task.npy
  y_three.npy
  trial_ids.npy
  meta.json
  train_X.npy / train_y_task.npy / …
  val_X.npy / val_y_task.npy / …
  split.json
```

常用参数：

| 参数 | 含义 |
|------|------|
| `--phases acquire` | 默认；`--phases all` 含 adapt/learn |
| `--no-filter` | 跳过 CAR/陷波/带通（调试） |
| `--no-split` | 不写 train_/val_ |
| `--out <dir>` | 自定义输出目录 |

---

## 4. 采一轮 → 预处理 → 训练（检查清单）

### A. 采集（须带 EEG）

- [ ] 用 `open_induction.bat` 或 `run_phase2_session`，**不要** `--no-acq`
- [ ] 真机：`--real --port COMx`；联调：默认 synthetic
- [ ] 完整走完适应 → 学习 → 准入(G) → 正式
- [ ] 会话目录存在：`eeg.csv`、`events.jsonl`、`session.meta.json`
- [ ] （可选）`python -m experiment_game.tools.verify_phase1_alignment --session <dir>`

### B. 切窗

- [ ] `run_phase4_epochs --session <dir>` 打印 `PHASE4_OK`
- [ ] `X.shape[1:] == (1, 8, 1000)`
- [ ] `y_task` 同时有 0 与 1；`y_three` 在有左右数据时含 1 与 2
- [ ] `meta.json` 中 `skipped` 可解释（多为时间越界）

### C. 训练（train_lab）

- [ ] 将 `DATA_DIR` 指向本会话的 `data/epochs/<名>/`（含 `train_X.npy` 等）
- [ ] 或把 npy 拷到 `preprocess_lab/out/...` 约定位置
- [ ] 运行既有 `train_task.py`（当前主任务头用 `y_task`）
- [ ] 确认 `n_chans=8`、`n_times=1000`

> 说明：synthetic 数据仅用于**流水线打通**，分类准确率无参考价值。

---

## 5. 验收（Phase 4）

| 项 | 通过条件 |
|----|----------|
| 形状 | `(N,1,8,1000)` |
| 标签 | `y_task∈{0,1}`，`y_three∈{0,1,2}`，且 Rest/Task 互相对应 |
| 过滤 | 默认不含 adapt/learn；reject trial 不进集 |
| 可训 | `train_X.npy` + `train_y_task.npy` 可被 Dataset 加载 |

---

## 6. 代码入口

| 模块 | 路径 |
|------|------|
| 加载会话 | `experiment_game/offline/load_session.py` |
| 切窗/标签 | `experiment_game/offline/epochs.py` |
| 主流程 | `experiment_game/offline/pipeline.py` |
| CLI | `experiment_game/tools/run_phase4_epochs.py` |
