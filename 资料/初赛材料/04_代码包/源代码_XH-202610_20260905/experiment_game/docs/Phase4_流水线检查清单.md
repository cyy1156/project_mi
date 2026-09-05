# Phase 4：采一轮 → 切窗 → 训练检查清单

> **版本**：v0.2  
> **日期**：2026-08-22  
> **配套**：`marker_spec.md`、`offline/`、`preprocess_lab`、`train_lab`

---

## 1. 目标

把 `experiment_game` 会话（`eeg.csv` + `events.jsonl`）切成与 `preprocess_lab` 一致的：

| 数组 | 形状 | 含义 |
|------|------|------|
| `X` | `(N, 1, 8, 500)` | 2 s × 250 Hz，8 通道（与 preprocess_lab 统一） |
| `y_task` | `(N,)` | 0=Rest，1=Task(左或右) |
| `y_three` | `(N,)` | 0=Rest，1=Left，2=Right |

切窗锚点：**`mi_start` / `rest_start`**（不用 `cue`）。在线范式可仍呈现 4 s MI/Rest；**离线只取起点起连续 2 s**（MI 等价 Cue+2~Cue+4）。  
默认只保留 **`phase == acquire`**，并丢弃 `trial_reject` 的 trial。

### 1.1 切窗模式（v0.2：固定窗 / 滑动窗）

| 参数 | 默认 | 说明 |
|------|------|------|
| `window_mode` | `fixed` | `fixed`=每阶段起点 1 窗；`slide`=阶段区间内滑切 |
| `win_sec` | 2.0 | 窗长（秒）→ `win_sec×250` 点；固定窗改窗长也用它 |
| `hop_ms` | 100 | 滑窗步长（毫秒），仅 slide 用 |

- **滑窗范围 = `[mi_start, mi_end)` / `[rest_start, rest_end)`**，窗必须完整落在阶段内——不会把保持段/过渡段切进训练窗；窗长超过阶段时长时该 trial 记 `window_exceeds_stage` 跳过。
- 参数入口：操作台 Setup「保存数据」组的切窗参数（自动切窗与 Summary 手动切窗的默认值）、Summary 页「一键 Phase4 切窗」旁的行内控件、CLI `--window-mode/--win-sec/--hop-ms`。
- 非默认参数输出到带后缀目录防覆盖：`<会话名>_slide_w2s_h100ms`；`meta.json` 记录 `window_mode/win_sec/hop_ms/baseline_s/n_times`。
- **训练注意**：滑窗样本之间高度相关，验证集必须**按 trial 划分**，不能按窗口随机划分（否则信息泄漏虚高）。`save_bundle` 的 train/val 是窗口级随机的，仅供快速冒烟；正式实验请按 `trial_ids.npy` 自行分组或用五折脚本。

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

在仓库根、使用根目录 `.venv`（需已 `pip install scipy`）：

```powershell
cd d:\cyy\MI

.\.venv\Scripts\python.exe `
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
- [ ] `X.shape[1:] == (1, 8, 500)`
- [ ] `y_task` 同时有 0 与 1；`y_three` 在有左右数据时含 1 与 2
- [ ] `meta.json` 中 `skipped` 可解释（多为时间越界）

### C. 训练（train_lab）

- [ ] 将会话 npy 接到约定位置（或拷到 `preprocess_lab/out/...`），保证有 `*_X.npy` / `*_y_task.npy` / `*_y_three.npy` / `*_subjects.npy`
- [ ] 现行入口：`code/train_lab/src/step/baselines_single/baseline_*.py`（例如 `python baseline_eegnet.py --data <tag>`；主任务头用 `y_task`，三分类独立训）
- [ ] 确认 `n_chans=8`、`n_times=500`
- [ ] 勿再把 `train_task_kfold.py` 当作现行入口（已在 `归档_旧训练入口/`，且不可直接跑）

> 说明：synthetic 数据仅用于**流水线打通**，分类准确率无参考价值。

---

## 5. 验收（Phase 4）

| 项 | 通过条件 |
|----|----------|
| 形状 | `(N,1,8,500)` |
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
