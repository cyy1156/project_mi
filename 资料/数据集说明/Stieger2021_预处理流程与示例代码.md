# Stieger 2021 数据集 —— 预处理流程与完整示例代码

> **性质**：实现说明文档。写清「怎么切、怎么标、代码放哪、示例怎么写」；**落地时新建专用模块，尽量不改 BCI2a 现有 steps / `pipeline.py`**。  
> **文献**：Stieger, Engel & He, *Scientific Data* (2021)  
> DOI：https://doi.org/10.1038/s41597-021-00883-1  
> **数据**：https://doi.org/10.6084/m9.figshare.13123148（亦见 NEMAR `nm000339`）  
> **对照**：`资料/数据集说明/数据集适配分析_扩展推荐.md` §1.3；目录结构见本文 §3（`src/datasets/{bci2a,stieger}` + `src/common`）  
> **注意**：本节结构为文档约定；**确认前不直接移动仓库代码**

---

## 0. 目标与原则

### 0.1 统一出口（与 BCI2a 相同 schema）

最终写出：

```text
code/preprocess_lab/out/stieger/
├── stieger_X.npy            # (N, 1, 8, 500) float32  # 现行：反馈段最后 2s
├── stieger_y_task.npy       # (N,)  0=静息, 1=任务
├── stieger_y_three.npy      # (N,)  0=空闲, 1=左, 2=右
├── stieger_subjects.npy     # (N,)  被试 ID，五折用
├── train_*.npy / val_*.npy  # 可选：试次级 8:2（基线用）
```

训练侧只认这套形状与标签；**Stieger 权重目录与 BCI2a 分开，从零训练，不做权重迁移对比。**

### 0.2 实现原则

| 原则 | 说明 |
|------|------|
| 按数据集分目录 | 2a / Stieger 各自放在 `src/datasets/<name>/`，互不混写标签与切窗 |
| 共用后处理 | 选 8 导、CAR、滤波、重采样、z-score、划分放在 `src/common/steps/` |
| 样本子集 | **LR+UD+2D**：左/右 + 原生「下」静息；双手 up 仍丢弃 |
| 变长试次 | 想象/反馈段最长约 **6 s**，分类窗取该段**最后 2 s**；不足 2 s **丢弃** |

---

## 1. 原始数据要点（官方）

### 1.1 规模与文件

| 项 | 内容 |
|----|------|
| 被试 | 62 人（S1–S62） |
| 会话 | 每人最多 11 个（含 baseline Session_1） |
| 文件名 | `SX_Session_Y.mat`（如 `S1_Session_1.mat`） |
| 每会话 | 约 450 trials，约 60 min EEG |
| 采样率 | **1000 Hz**（`BCI.SRATE`） |
| 通道 | **62** 导（10–10），名在 `BCI.chaninfo.labels` |

建议本机目录：

```text
DATA/stieger/
├── S1_Session_1.mat
├── S1_Session_2.mat
├── ...
└── S62_Session_11.mat
```

### 1.2 `BCI` 结构体关键字段

| 字段 | 含义 |
|------|------|
| `data` | `1×450` cell，每格 `(nChannels×nTime)`，单位 μV |
| `time` | 每 trial 时间轴（ms），相对**目标呈现**；t=-2000 起 ITI |
| `SRATE` | 恒为 1000 |
| `TrialData` | 试次元数据（见下表） |
| `chaninfo.labels` | 62 通道名 |
| `chaninfo.noisechan` | 会话级噪声通道（可选插值/排除） |

### 1.3 试次时序（官方）

```text
样本索引（约）:  1 ──────── 2001 ──────── 4001 ────── resultind ── +1000
时间 t (ms):   -2000        0            2000
阶段:          ITI 2s    目标呈现 2s    反馈控制 ≤6s      结束后 1s
```

- 反馈起点：约 **index=4001**（t=2000 ms），官方整段反馈：  
  `trial_feedback = BCI.data{trial}(:, 4001:TrialData(trial).resultind)`
- 反馈/想象长度变长：约 0.04–6.04 s；timeout 常顶满约 **6 s**
- **本项目分类窗**：不取反馈开头，而取该反馈段的**最后 2 s**（满 6 s 时即第 4–6 s）

### 1.4 范式与标签（官方）

**被试指令（Methods）：**

> 想左/右手开合 → 光标左/右；想**双手**开合 → 光标**上**；光标**下** → **voluntarily rest / clear your mind**（主动静息）。

**`TrialData` 字段：**

| 字段 | 取值 |
|------|------|
| `tasknumber` | 1=LR，2=UD，3=2D |
| `targetnumber` | 1=right，2=left，3=up，4=down |
| `artifact` | 1=含伪迹，0=无 |
| `resultind` | 反馈结束样本索引 |
| `triallength` | 反馈时长（秒） |

> 注意：表里写的是 `4=down`，不是字面 `rest`；按 Methods，**down = 主动静息**。

### 1.5 本项目标签映射（工程约定）

| `targetnumber` | 官方含义 | 本项目处理 |
|----------------|----------|------------|
| 2 | left | `y_task=1`, `y_three=1` |
| 1 | right | `y_task=1`, `y_three=2` |
| 3 | up（双手） | **丢弃** |
| 4 | down（主动静息） | `y_task=0`, `y_three=0` |

与 BCI2a 对比：2a 是 `1=左, 2=右`；Stieger 是 `1=右, 2=左`——**绝不能直接套 `THREE_MAP={1:1,2:2}`**。

---

## 2. 预处理决策（已拍板）

| # | 决策项 | 取值 |
|---|--------|------|
| 1 | 窗位置 | 想象/反馈段最长约 **6 s**，取该段**最后 2 s**（对齐反馈终点 `resultind` 往前切） |
| 2 | 窗长 | 固定 **2.0 s**（→ 500@250Hz，与 BCI2a 统一） |
| 3 | 短于 2 s | 反馈时长 `< 2 s` **丢弃**，并统计比例 |
| 4 | 基线 | 分类窗起点前 **0.5 s** 均值校正（通常落在 6 s 想象段的前部） |
| 5 | 静息来源 | **原生 `targetnumber==4`（down）**，来自 **UD + 2D**；不用 ITI 人造静息 |
| 6 | 范式范围 | **`tasknumber∈{1,2,3}`**，按块保留左/右/静息（见下表）；**始终丢弃 up（双手）** |
| 7 | 伪迹 | `artifact==1` 的 trial 丢弃 |
| 8 | 降采样 | 1000 Hz → **250 Hz**（2 s → **500** 点） |
| 9 | 通道 | 固定序：`Cz, C3, C4, CP3, FC4, FC3, CP4, CPz` |
| 10 | 滤波 | CAR → 50 Hz notch → 8–30 Hz bandpass（与 2a 相同） |

各 `tasknumber` 保留的 `targetnumber`：

| task | 块 | 保留 target | 丢弃 |
|------|-----|-------------|------|
| 1 | LR | 1=右，2=左 | 3、4（LR 中通常本无） |
| 2 | UD | **4=下（静息）** | 3=上（双手） |
| 3 | 2D | 1=右，2=左，**4=下（静息）** | 3=上（双手） |

> 这样双头标签可同时有左 / 右 / 静息；仅跑 LR 时 `y_task` 会全是任务态（无静息）。

满 6 s 的 timeout 试次示意：

```text
反馈/想象 [0 ──────────────── 6]s
分类窗              [4 ────── 6]s   ← 最后 2s
基线             [1.5–2]s
```

若试次提前命中/失败（反馈只有 4–6 s），仍取**实际反馈段的最后 2 s**。

### 2.1 与 BCI2a 流水线对照

```text
BCI2a（现行）:
  连续流 → ContinuousEEG → 选8导 → 滤波
  → 滤 y∈{1,2} → cue 后 2–4s → ITI 造静息（Cue 前 2s）
  →（已是 250Hz）→ zscore → (N,1,8,500) npy

Stieger:
  trial cell →【专用】过滤范式/伪迹/标签/定窗
  →【复用】选8导 → 滤波
  →【复用】1000→250、zscore、拼张量 → npy
```

---

## 3. 目标目录结构（文档约定；落地前先按此改，暂不直接挪代码）

> **原则**：同一个大文件夹 `src/datasets/` 下，按数据集分子文件夹；共用逻辑进 `src/common/`。  
> **本文以下所有示例路径 / import 均按此目标结构书写。**  
> **仓库状态（已物理迁移）**：`common/`、`datasets/bci2a/`、`datasets/stieger/` 已按对照表搬迁并更新 import。  
> **仓库状态**：`src/datasets/stieger/batch.py` 已按 §5.8 落地；默认输出 `out/stieger_2s`（2s/500）。

相对仓库根目录 `MI/`：

```text
code/preprocess_lab/
├── config/
│   ├── bci2a.yaml                       # 2a 批处理配置
│   └── stieger.yaml                     # Stieger 批处理配置
│
├── src/
│   ├── common/                          # ★ 共用（两数据集都调用，不含特定标签语义）
│   │   ├── __init__.py
│   │   ├── eeg_types.py                 # ContinuousEEG 等（原 src/eeg_types.py）
│   │   ├── config_load.py               # 读 yaml
│   │   └── steps/
│   │       ├── __init__.py
│   │       ├── select_channels.py       # 固定 8 导
│   │       ├── filter_car.py            # CAR + notch + bandpass
│   │       ├── resample_zscore.py       # 重采样 / z-score / 拼张量
│   │       ├── epoch_baseline.py        # 2a 切窗用（cue 后 2–4s）；Stieger 不用
│   │       └── split_subjects.py        # 8:2 / 五折辅助
│   │
│   └── datasets/                        # ★ 大文件夹：每个数据集一个小文件夹
│       ├── __init__.py
│       ├── registry.py                  # 注册 load_bci2a_mat / load_stieger_mat
│       │
│       ├── bci2a/                       # ★ BCI IV 2a 专用
│       │   ├── __init__.py
│       │   ├── load_mat.py              # 原 io/load_bci2a_mat.py
│       │   ├── labels.py                # 原 steps/harmonize_labels.py（2a 标签+造静息）
│       │   ├── pipeline.py              # 原 src/pipeline.py
│       │   └── batch.py                 # 原 src/pipline_batch.py（服务 2a）
│       │
│       └── stieger/                     # ★ Stieger 2021 专用
│           ├── __init__.py
│           ├── load_mat.py              # 读 SX_Session_Y.mat
│           ├── labels.py                # targetnumber → 双头标签
│           ├── paradigm.py              # LR 过滤 / 伪迹 / 短窗
│           ├── windows.py               # 反馈段最后 2s + 基线
│           ├── pipeline.py              # 单会话串联
│           └── batch.py                 # 增量批处理（manifest 防重复）
│
└── out/
    ├── bci2a/                           # 2a 输出 npy
    └── stieger/                         # Stieger 输出 npy + manifest / log

DATA/
├── bci2a/                               # 原始 2a .mat
└── stieger/                             # 原始 Stieger .mat（可分批下载）

资料/数据集说明/
└── Stieger2021_预处理流程与示例代码.md   # 本文
```

### 3.1 职责划分

| 位置 | 放什么 | 不放什么 |
|------|--------|----------|
| `src/common/steps/` | 与数据集无关的信号处理、张量、划分 | 2a/Stieger 的事件码、范式过滤 |
| `src/datasets/bci2a/` | 2a 读入、标签、造静息（Cue前2s）、cue+2~4s 切窗、批处理 | Stieger 字段或切窗 |
| `src/datasets/stieger/` | Stieger 读入、标签、范式、最后 2s 窗、增量 batch | 2a 的 `THREE_MAP` / cue 切窗 |
| `src/datasets/registry.py` | 按名字取 loader | 业务预处理逻辑 |
| `config/*.yaml` | 路径、是否造静息、增量开关等 | 复杂 Python 逻辑 |
| `out/<dataset>/` | 各库自己的 npy / 清单 | 混用对方权重或清单 |

### 3.2 旧路径 → 新路径对照（落地迁移用）

| 现状（旧） | 目标（新） |
|------------|------------|
| `src/eeg_types.py` | `src/common/eeg_types.py` |
| `src/config_load.py` | `src/common/config_load.py` |
| `src/steps/*.py` | `src/common/steps/*.py` |
| `src/io/registry.py` | `src/datasets/registry.py` |
| `src/io/load_bci2a_mat.py` | `src/datasets/bci2a/load_mat.py` |
| `src/steps/harmonize_labels.py` | `src/datasets/bci2a/labels.py` |
| `src/pipeline.py` | `src/datasets/bci2a/pipeline.py` |
| `src/pipline_batch.py` | `src/datasets/bci2a/batch.py` |
| `src/io/stieger/*` | `src/datasets/stieger/*`（同名文件） |
| `src/pipeline_stieger.py` | `src/datasets/stieger/pipeline.py` |
| `src/pipeline_stieger_batch.py`（原先仓库中无此文件） | `src/datasets/stieger/batch.py`（**已实现**，§5.8） |

### 3.3 import / 运行方式（目标）

```python
# Stieger
from src.datasets.stieger.load_mat import load_stieger_mat, StiegerTrial
from src.datasets.stieger.labels import map_target
from src.common.steps.select_channels import select_channels

# BCI2a
from src.datasets.bci2a.load_mat import load_bci2a_mat
from src.datasets.bci2a.labels import filter_left_right_events
```

```bash
cd code/preprocess_lab

# 2a
python -m src.datasets.bci2a.pipeline
python -m src.datasets.bci2a.batch --cfg config/bci2a.yaml

# Stieger
python -m src.datasets.stieger.pipeline
python -m src.datasets.stieger.batch
python -m src.datasets.stieger.batch --delete-raw
```

> 草稿 loader（`load_gdf` / `load_openbci_csv`）落地时可放到 `src/datasets/_draft/` 或各自数据集文件夹；**正式两库先只维护 `bci2a/` 与 `stieger/`。**

---


## 4. 端到端流程（逐步）

```text
SX_Session_Y.mat
    │
    ▼  Step A  load_mat
       解析 BCI；得到每 trial: X(n_ch, n_t), time, TrialData, ch_names, fs=1000
    │
    ▼  Step B  paradigm filter
       task 1 (LR)：保留 target ∈ {1,2}（右/左）
       task 2 (UD)：保留 target == 4（下=静息）
       task 3 (2D)：保留 target ∈ {1,2,4}（右/左/静息）
       一律丢弃 target==3（双手 up）、artifact==1、反馈 <2s
    │
    ▼  Step C  通道 + 滤波（在 trial 矩阵上，转置为 (T,C)）
       选 8 导 → CAR → notch 50 → bandpass 8–30
       （也可先拼会话再滤；示例用逐 trial，实现简单）
    │
    ▼  Step D  windows
       fb0 = 反馈起点（time 上 t>=2000 ms 的首点；约 index 4000）
       fb1 = 反馈终点（优先 TrialData.resultind，转为半开区间终点）
       分类窗 = [fb1 - 2s, fb1)     # 想象段最后 2s（满 6s 时即 4–6s）
       基线   = [窗起点 - 0.5s, 窗起点)
       若反馈不足 2s，或基线越界 → 丢弃
    │
    ▼  Step E  labels
       2→(1,1) 左；1→(1,2) 右；4→(0,0) 静息
    │
    ▼  Step F  resample + zscore
       (2000,8)@1000Hz → (500,8)@250Hz → 每通道 z-score
    │
    ▼  Step G  汇总
       stack → (N,1,8,500)；写 subjects；可选 8:2 split
```

---

## 5. 完整示例代码

以下代码为**可直接粘贴落地**的完整示例；依赖与现工程一致：`numpy`、`scipy`、`mne`、`pyyaml`。

### 5.1 `config/stieger.yaml`

**路径：** `code/preprocess_lab/config/stieger.yaml`

```yaml
# Stieger 2021 —— 整库批处理（LR 左右 + UD/2D 静息 + 2D 左右）
dataset: stieger
loader: load_stieger_mat

# 按本机路径修改
data_glob: "D:/360MoveData/Users/ckgxnn/Desktop/MI/DATA/stieger/S*_Session_*.mat"

# 调试时可改为只列几个文件：
# data_files:
#   - "D:/360MoveData/Users/ckgxnn/Desktop/MI/DATA/stieger/S1_Session_1.mat"

subject_from: stieger_stem   # S12_Session_3.mat → "S12"

# 范式：1=LR，2=UD，3=2D —— 收集全部可用的左/右/静息
use_tasks: [1, 2, 3]
# 各 task 允许的 target 见 paradigm.ALLOWED_TARGETS_BY_TASK
drop_targets: [3]       # 双手 up 全局丢弃
rest_targets: [4]       # down = 原生静息（主要来自 UD/2D）
make_rest_from_iti: false

# 切窗：想象/反馈最长约 6s，取最后 2s
feedback_t_ms: 2000     # 反馈起点（相对目标呈现，仅用于定位 fb0）
win_sec: 2.0            # 从反馈终点往前取
baseline_sec: 0.5       # 紧贴分类窗前
min_feedback_sec: 2.0   # 反馈不足 2s 则丢

target_chans: [Cz, C3, C4, CP3, FC4, FC3, CP4, CPz]
fs_out: 250

val_ratio: 0.2
seed: 42

out_dir: "out/stieger_2s"
save_full: true
save_split: true

# ---- 增量流水（分批下载必备）----
incremental: true                 # 已有 npy 则追加，不覆盖
manifest_name: "processed_manifest.json"
batch_log_name: "batch_log.jsonl"
skip_if_processed: true           # 清单里已有则跳过（防重复）
delete_raw_after_ok: false        # true=该文件处理成功并落盘后删除 .mat（慎用）
rebuild_split: false              # false=只把本批新样本划分并追加到 train/val
                                  # true=每次用全量重划 8:2（旧样本归属会变，不推荐）
```

---

### 5.2 `src/datasets/stieger/__init__.py`

**路径：** `code/preprocess_lab/src/datasets/stieger/__init__.py`

```python
from src.datasets.stieger.load_mat import load_stieger_mat, StiegerTrial

__all__ = ["load_stieger_mat", "StiegerTrial"]
```

---

### 5.3 `src/datasets/stieger/labels.py`

**路径：** `code/preprocess_lab/src/datasets/stieger/labels.py`

```python
"""Stieger targetnumber → 本项目双头标签。"""
from __future__ import annotations

# 官方: 1=right, 2=left, 3=up, 4=down(rest)
# 本项目 y_three: 1=左, 2=右, 0=静息
TARGET_TO_LABELS: dict[int, tuple[int, int]] = {
    2: (1, 1),  # left  → task, left
    1: (1, 2),  # right → task, right
    4: (0, 0),  # down  → rest
}
DROP_TARGETS = {3}  # both-hands up


def map_target(targetnumber: int) -> tuple[int, int] | None:
    """返回 (y_task, y_three)；应丢弃则返回 None。"""
    t = int(targetnumber)
    if t in DROP_TARGETS:
        return None
    return TARGET_TO_LABELS.get(t)
```

---

### 5.4 `src/datasets/stieger/paradigm.py`

**路径：** `code/preprocess_lab/src/datasets/stieger/paradigm.py`

> **现行规则（请同步改你本地的 `paradigm.py`；本文不直接改仓库代码）**  
> - `use_tasks=(1,2,3)`：LR + UD + 2D  
> - 按块限制 target，拿到**所有可用的左 / 右 / 静息**，仍丢双手 up  

```python
"""范式与质量过滤：LR/UD/2D 中收集左、右、静息。"""
from __future__ import annotations

from src.datasets.stieger.labels import map_target

# 各 tasknumber 允许保留的 targetnumber
# 1=right, 2=left, 3=up(双手→丢), 4=down(静息)
ALLOWED_TARGETS_BY_TASK: dict[int, set[int]] = {
    1: {1, 2},       # LR：右 / 左
    2: {4},          # UD：仅下=静息（上=双手丢）
    3: {1, 2, 4},    # 2D：右 / 左 / 静息
}

DEFAULT_USE_TASKS: tuple[int, ...] = (1, 2, 3)


def keep_trial(
    tasknumber: int,
    targetnumber: int,
    artifact: int | float,
    triallength: float,
    *,
    use_tasks: tuple[int, ...] = DEFAULT_USE_TASKS,
    min_feedback_sec: float = 2.0,
    allowed_by_task: dict[int, set[int]] | None = None,
) -> bool:
    """
    是否保留该试次。
    - 非 use_tasks → 丢
    - artifact==1 → 丢
    - 反馈时长 < min_feedback_sec → 丢
    - target 不在该 task 允许集合，或 map_target 失败（如双手）→ 丢
    """
    task = int(tasknumber)
    target = int(targetnumber)
    allowed_map = allowed_by_task or ALLOWED_TARGETS_BY_TASK

    if task not in use_tasks:
        return False
    if artifact is not None and int(artifact) == 1:
        return False
    if float(triallength) < min_feedback_sec:
        return False

    allowed = allowed_map.get(task)
    if allowed is None or target not in allowed:
        return False
    if map_target(target) is None:
        return False
    return True
```

**与旧版（仅 LR）对比：**

| | 旧 | 现行 |
|--|----|------|
| `use_tasks` | `(1,)` | `(1, 2, 3)` |
| 静息 | LR 中几乎没有 → `y_task` 常全为 1 | UD/2D 的 `target==4` |
| 左右 | 仅 LR | LR + 2D 的左右 |

---

### 5.5 `src/datasets/stieger/windows.py`

**路径：** `code/preprocess_lab/src/datasets/stieger/windows.py`

> **切窗规则（已定稿，勿用旧版）**  
> - ❌ 旧版：从反馈起点 `fb0` 往后取 4 s（`x[fb0:fb0+4s]`）  
> - ✅ **现行**：想象/反馈最长约 6 s，取该段**最后 2 s**（`x[fb1-2s:fb1]`），基线为分类窗前 0.5 s  
> - 调用时必须传入 `resultind=tr.resultind`（见 §5.7）

```python
"""想象/反馈段取最后 win_sec 秒 + 基线校正（现行默认 2s）。"""
from __future__ import annotations

import numpy as np


def feedback_start_index(time_ms: np.ndarray, feedback_t_ms: float = 2000.0) -> int:
    """
    time_ms: 1×nTime，相对目标呈现（ms）。
    返回第一个 t >= feedback_t_ms 的样本下标（0-based）。
    """
    t = np.asarray(time_ms).reshape(-1)
    idx = np.where(t >= feedback_t_ms - 1e-6)[0]
    if len(idx) == 0:
        # 回退：官方文档常用 4001（1-based）→ 0-based 4000
        return 4000
    return int(idx[0])


def feedback_end_index(
    n_times: int,
    resultind: int,
    fb0: int,
) -> int:
    """
    反馈终点（Python 半开区间右端）。
    官方 resultind 为 MATLAB 1-based 结束下标时，半开终点数值上等于 resultind。
    异常则回退 n_times。
    """
    end = int(resultind)
    end = min(max(end, fb0), n_times)
    if end <= fb0:
        end = n_times
    return int(end)


def extract_mi_or_rest_window(
    x_tc: np.ndarray,
    time_ms: np.ndarray,
    fs: float,
    *,
    resultind: int,
    feedback_t_ms: float = 2000.0,
    win_sec: float = 2.0,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    """
    x_tc: (n_times, n_ch)

    想象/反馈最长约 6s：分类窗取反馈段【最后 win_sec 秒】，
    即 [fb_end - win_sec, fb_end)。
    基线 = 分类窗起点前 baseline_sec。
    返回 (win_sec*fs, n_ch)；不够长则 None。

    满 6s 示意：
      反馈 [0 -------- 6]s
      基线          [1.5–2]
      分类窗               [4 -- 6]  ← 最后 2s
    """
    fb0 = feedback_start_index(time_ms, feedback_t_ms)
    fb1 = feedback_end_index(x_tc.shape[0], resultind, fb0)
    n_base = int(round(baseline_sec * fs))
    n_win = int(round(win_sec * fs))

    if fb1 - fb0 < n_win:
        return None  # 反馈不足 win_sec

    win_start = fb1 - n_win
    base_start = win_start - n_base
    if base_start < 0:
        return None

    base = x_tc[base_start:win_start].mean(axis=0, keepdims=True)
    win = x_tc[win_start:fb1] - base
    if win.shape[0] != n_win:
        return None
    return win.astype(np.float64)
```

---

### 5.6 `src/datasets/stieger/load_mat.py`

**路径：** `code/preprocess_lab/src/datasets/stieger/load_mat.py`

```python
"""读取单个 Stieger 会话 .mat → list[StiegerTrial]。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


@dataclass
class StiegerTrial:
    """单试次中间表示（尚未选导/滤波）。"""
    subject: str
    session: str
    trial_index: int          # 0-based
    x: np.ndarray             # (n_times, n_channels) μV
    time_ms: np.ndarray       # (n_times,)
    fs: float
    ch_names: list[str]
    tasknumber: int
    targetnumber: int
    artifact: int
    triallength: float
    resultind: int            # 1-based 或与 MATLAB 一致的结束索引


def _parse_subject_session(path: Path) -> tuple[str, str]:
    # S12_Session_3.mat → ("S12", "Session_3")
    m = re.match(r"(S\d+)_Session_(\d+)", path.stem, flags=re.IGNORECASE)
    if not m:
        return path.stem, "unknown"
    return f"S{int(m.group(1)[1:])}", f"Session_{int(m.group(2))}"


def _as_str_list(labels_obj) -> list[str]:
    """兼容 MATLAB cellstr / 嵌套 object。"""
    out: list[str] = []
    arr = np.asarray(labels_obj, dtype=object).reshape(-1)
    for item in arr:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="ignore"))
        else:
            # 常见: array(['C3'], dtype='<U2') 或再包一层
            s = np.asarray(item).reshape(-1)
            out.append(str(s[0]) if len(s) else str(item))
    return out


def _get_field(trialdata_i, name: str, default=None):
    """TrialData 可能是 struct 数组或 mat_struct。"""
    if isinstance(trialdata_i, np.void) or hasattr(trialdata_i, "_fieldnames"):
        try:
            return trialdata_i[name]
        except Exception:
            return getattr(trialdata_i, name, default)
    if isinstance(trialdata_i, dict):
        return trialdata_i.get(name, default)
    return default


def load_stieger_mat(mat_path: Path | str) -> list[StiegerTrial]:
    """
    读一个会话文件，返回全部试次（未过滤）。
    注意: scipy 对嵌套 struct 较挑；若字段取不到，用 mat73 / hdf5storage 再包一层。
    """
    mat_path = Path(mat_path)
    subject, session = _parse_subject_session(mat_path)

    raw = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "BCI" not in raw:
        raise KeyError(f"{mat_path.name} 中无 BCI 变量，键={list(raw)}")
    bci = raw["BCI"]

    fs = float(np.asarray(bci.SRATE).reshape(-1)[0])
    ch_names = _as_str_list(bci.chaninfo.labels)

    data_cells = np.asarray(bci.data, dtype=object).reshape(-1)
    time_cells = np.asarray(bci.time, dtype=object).reshape(-1)
    trialdata = np.asarray(bci.TrialData).reshape(-1)

    n = len(data_cells)
    trials: list[StiegerTrial] = []
    for i in range(n):
        x = np.asarray(data_cells[i], dtype=np.float64)
        # 官方: nChannels × nTime → 转为 (T, C)
        if x.ndim != 2:
            continue
        if x.shape[0] == len(ch_names) or x.shape[0] < x.shape[1]:
            x = x.T
        t_ms = np.asarray(time_cells[i], dtype=np.float64).reshape(-1)
        td = trialdata[i]

        def _scalar(name, default=0):
            v = _get_field(td, name, default)
            if v is None:
                return default
            a = np.asarray(v).reshape(-1)
            return a[0] if len(a) else default

        trials.append(
            StiegerTrial(
                subject=subject,
                session=session,
                trial_index=i,
                x=x,
                time_ms=t_ms,
                fs=fs,
                ch_names=ch_names,
                tasknumber=int(_scalar("tasknumber", -1)),
                targetnumber=int(_scalar("targetnumber", -1)),
                artifact=int(_scalar("artifact", 0)),
                triallength=float(_scalar("triallength", 0.0)),
                resultind=int(_scalar("resultind", x.shape[0])),
            )
        )
    return trials
```

> **读 mat 提示**：若本机 `scipy.io.loadmat` 打不开该 figshare 文件（偶发 v7.3/HDF5），改用：
>
> ```python
> import mat73
> raw = mat73.loadmat(str(mat_path))
> bci = raw["BCI"]  # 此时多为 dict
> ```
>
> 并相应把 `bci.SRATE` 改成 `bci["SRATE"]` 等字典访问。可在 `src/datasets/stieger/load_mat.py` 里做 try/except 双路径。

---

### 5.7 `src/datasets/stieger/pipeline.py`（串联 + 验收）

**路径：** `code/preprocess_lab/src/datasets/stieger/pipeline.py`

```python
"""Stieger 专用预处理流水线（位于 datasets/stieger/，与 bci2a 分离）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.stieger.load_mat import load_stieger_mat, StiegerTrial
from src.datasets.stieger.labels import map_target
from src.datasets.stieger.paradigm import keep_trial, ALLOWED_TARGETS_BY_TASK
from src.datasets.stieger.windows import extract_mi_or_rest_window
from src.common.steps.select_channels import select_channels, TARGET_CHANNELS
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.resample_zscore import (
    resample_to_1000,
    trial_zscore,
    to_model_tensor,
)


def _process_one_trial(tr: StiegerTrial) -> tuple[np.ndarray, int, int] | None:
    if not keep_trial(
        tr.tasknumber,
        tr.targetnumber,
        tr.artifact,
        tr.triallength,
        use_tasks=(1, 2, 3),  # LR + UD + 2D
        min_feedback_sec=2.0,
    ):
        return None

    labels = map_target(tr.targetnumber)
    if labels is None:
        return None
    y_task, y_three = labels

    # 通道可能因命名大小写不一致：统一 strip
    ch_names = [c.strip() for c in tr.ch_names]
    try:
        x = select_channels(tr.x, ch_names)
    except KeyError:
        # 尝试常见别名（若官方标签带空格等）
        alias = {n.upper(): n for n in ch_names}
        mapped = []
        for want in TARGET_CHANNELS:
            key = want.upper()
            if key not in alias:
                return None
            mapped.append(ch_names.index(alias[key]))
        x = tr.x[:, mapped]

    x = car_reference(x)
    x = notch_and_bandpass(x, tr.fs)

    win = extract_mi_or_rest_window(
        x,
        tr.time_ms,
        tr.fs,
        resultind=tr.resultind,
        feedback_t_ms=2000.0,
        win_sec=2.0,
        baseline_sec=0.5,
    )
    if win is None or win.shape[0] != int(2.0 * tr.fs):
        return None

    win = resample_to_1000(win, fs_in=tr.fs, fs_out=250.0, win_sec=2.0)
    if win.shape != (500, 8):
        return None
    win = trial_zscore(win)
    return win, int(y_task), int(y_three)


def preprocess_session(
    mat_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    单会话 → X, y_task, y_three, subjects, stats
    """
    mat_path = Path(mat_path)
    trials = load_stieger_mat(mat_path)
    xs, yt, y3, sids = [], [], [], []
    stats = {
        "n_raw": len(trials),
        "n_keep": 0,
        "n_drop_short": 0,
        "n_drop_artifact": 0,
        "n_drop_target": 0,
        "n_drop_task": 0,
    }

    for tr in trials:
        # 统计用：与 keep_trial 同一套规则（含 task 2/3）
        if int(tr.tasknumber) not in (1, 2, 3):
            stats["n_drop_task"] += 1
            continue
        if int(tr.artifact) == 1:
            stats["n_drop_artifact"] += 1
            continue
        # 该 task 不允许的 target（含双手 up、UD 的左右等）
        allowed = ALLOWED_TARGETS_BY_TASK.get(int(tr.tasknumber), set())
        if int(tr.targetnumber) not in allowed or map_target(tr.targetnumber) is None:
            stats["n_drop_target"] += 1
            continue
        if float(tr.triallength) < 2.0:
            stats["n_drop_short"] += 1
            continue

        out = _process_one_trial(tr)
        if out is None:
            stats["n_drop_short"] += 1
            continue
        win, y_task, y_three = out
        xs.append(win)
        yt.append(y_task)
        y3.append(y_three)
        sids.append(tr.subject)

    stats["n_keep"] = len(xs)
    if not xs:
        empty = np.zeros((0, 1, 8, 500), np.float32)
        z = np.zeros((0,), np.int64)
        return empty, z, z.copy(), np.array([], dtype=object), stats

    X = to_model_tensor(xs)
    return (
        X,
        np.asarray(yt, dtype=np.int64),
        np.asarray(y3, dtype=np.int64),
        np.asarray(sids, dtype=object),
        stats,
    )


def sanity_check_outputs(X, y_task, y_three) -> None:
    assert len(X) > 0, "没有有效试次"
    assert X.ndim == 4 and X.shape[1:] == (1, 8, 500)
    assert len(X) == len(y_task) == len(y_three)
    assert set(np.unique(y_task)).issubset({0, 1})
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    assert np.all((y_three == 0) == (y_task == 0))
    assert np.all(y_task[y_three > 0] == 1)
    assert np.isfinite(X).all()
    print(
        "OK",
        "X", X.shape,
        "y_task", np.bincount(y_task, minlength=2),
        "y_three", np.bincount(y_three, minlength=3),
    )


def main() -> None:
    # 调试：先跑一个会话
    mat_path = Path(
        r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\stieger\S1_Session_1.mat"
    )
    out_dir = Path(__file__).resolve().parents[1] / "out" / "stieger"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_task, y_three, subjects, stats = preprocess_session(mat_path)
    print("stats:", stats)
    sanity_check_outputs(X, y_task, y_three)

    np.save(out_dir / "debug_S1S1_X.npy", X)
    np.save(out_dir / "debug_S1S1_y_task.npy", y_task)
    np.save(out_dir / "debug_S1S1_y_three.npy", y_three)
    np.save(out_dir / "debug_S1S1_subjects.npy", subjects)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
```

---

### 5.8 增量批处理（分批下载 → 处理 → 可删原数据 → 再下一批）

**路径：** `code/preprocess_lab/src/datasets/stieger/batch.py`  
（与 `src/datasets/bci2a/batch.py` 并列；互不共用入口）

#### 5.8.1 为什么必须增量

Stieger 全库约 598 个会话、体量很大，典型操作是：

```text
下载一批 .mat
  → 预处理并【追加】到 out/stieger/*.npy
  → 写入 processed_manifest（防重复）
  → （可选）删除本批原始 .mat 腾磁盘
  → 再下载下一批，重复
```

若每次全量覆盖保存，删掉的原始数据无法重跑，且会丢掉上一批结果。

#### 5.8.2 落盘文件约定

```text
out/stieger_2s/                # 现行默认（2s → 500 点；旧文档曾写 out/stieger）
├── stieger_X.npy              # 全量特征（只追加）
├── stieger_y_task.npy
├── stieger_y_three.npy
├── stieger_subjects.npy
├── train_*.npy / val_*.npy    # 默认：只追加本批新划分；正式五折主要用全量+subjects
├── processed_manifest.json    # ★ 已处理文件清单（防重复的唯一真相源）
└── batch_log.jsonl            # 每次运行一条日志（可追溯）
```

**清单主键：** 文件名 stem，如 `S1_Session_1`（同一会话不应下两次；若文件更新可加 `size`/`sha1` 校验）。

#### 5.8.3 推荐日常流程

1. 把本批 `.mat` 放进 `DATA/stieger/`  
2. 跑 `python -m src.datasets.stieger.batch`  
3. 看终端：`skip`（已在清单）/ `ok`（新处理）/ `fail`  
4. 确认 `stieger_X.npy` 的 N 增加、`manifest` 有新条目  
5. 删除本批原始 `.mat`（或设 `delete_raw_after_ok: true`）  
6. 下载下一批，重复 1–5  
7. 全部跑完后，用 `stieger_*.npy` + `subjects` 做被试五折训练  

> **五折训练**读全量 `stieger_X/y_*/subjects`，不依赖 `train_/val_` 是否完美。  
> `train_/val_` 仅给试次混合基线用；增量时用「只划分本批并追加」，避免每次重划打乱旧样本归属。

#### 5.8.4 完整示例代码

```python
"""Stieger 增量批处理：跳过已处理 → 追加 npy → 写清单/日志 → 可选删原文件。"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.stieger.pipeline import preprocess_session, sanity_check_outputs
from src.common.steps.split_subjects import split_all_trials

FULL_KEYS = ("X", "y_task", "y_three", "subjects")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_id(path: Path) -> str:
    """清单主键：会话文件名（不含扩展名）。"""
    return path.stem  # S1_Session_1


def file_fingerprint(path: Path) -> dict:
    """辅助校验：大小 + 快速采样 hash（整文件 sha 太慢时可只用 size）。"""
    st = path.stat()
    h = hashlib.sha1()
    with path.open("rb") as f:
        # 只读头尾各 1MB，大文件也快；若需绝对严谨可改全文 sha1
        head = f.read(1 << 20)
        if st.st_size > (1 << 20):
            f.seek(max(0, st.st_size - (1 << 20)))
            tail = f.read(1 << 20)
        else:
            tail = b""
    h.update(head)
    h.update(tail)
    return {
        "size": int(st.st_size),
        "sha1_sample": h.hexdigest(),
        "mtime": int(st.st_mtime),
    }


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "files": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _npy_path(out_dir: Path, kind: str, split: str | None = None) -> Path:
    if split is None:
        return out_dir / f"stieger_{kind}.npy"
    return out_dir / f"{split}_{kind}.npy"


def load_existing_full(out_dir: Path) -> dict[str, np.ndarray] | None:
    paths = {k: _npy_path(out_dir, k) for k in FULL_KEYS}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: np.load(p, allow_pickle=(k == "subjects")) for k, p in paths.items()}


def save_full(out_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, arr in arrays.items():
        np.save(_npy_path(out_dir, k), arr)


def concat_or_new(
    old: dict[str, np.ndarray] | None,
    new: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if old is None:
        return new
    return {k: np.concatenate([old[k], new[k]], axis=0) for k in FULL_KEYS}


def append_split_arrays(
    out_dir: Path,
    split: str,
    X, y_task, y_three, subjects,
) -> None:
    """把本批划分结果追加到已有 train_/val_；没有则新建。"""
    keys = ("X", "y_task", "y_three", "subjects")
    new = {
        "X": X,
        "y_task": y_task,
        "y_three": y_three,
        "subjects": subjects,
    }
    for k in keys:
        p = _npy_path(out_dir, k, split=split)
        if p.exists():
            old = np.load(p, allow_pickle=(k == "subjects"))
            arr = np.concatenate([old, new[k]], axis=0)
        else:
            arr = new[k]
        np.save(p, arr)


def run_incremental_batch(
    data_glob: str,
    out_dir: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    skip_if_processed: bool = True,
    delete_raw_after_ok: bool = False,
    rebuild_split: bool = False,
    manifest_name: str = "processed_manifest.json",
    batch_log_name: str = "batch_log.jsonl",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / manifest_name
    log_path = out_dir / batch_log_name
    manifest = load_manifest(manifest_path)

    files = sorted(Path(p) for p in glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"未匹配到文件: {data_glob}")

    batch_id = _utc_now()
    n_skip = n_ok = n_fail = n_empty = 0
    new_Xs, new_yts, new_y3s, new_sids = [], [], [], []

    print(f"[batch {batch_id}] 扫描到 {len(files)} 个文件；清单已有 {len(manifest['files'])} 条")

    for fp in files:
        fid = file_id(fp)
        fp_info = file_fingerprint(fp)

        # ---- 防重复：清单命中则跳过 ----
        if skip_if_processed and fid in manifest["files"]:
            prev = manifest["files"][fid]
            # 同名但大小变了：警告并仍跳过（避免静默污染）；要重跑请先从清单删除该键
            if prev.get("size") and prev["size"] != fp_info["size"]:
                print(f"  WARN skip {fp.name}: 清单已有但 size 不同 prev={prev.get('size')} now={fp_info['size']}")
            else:
                print(f"  skip {fp.name} (already in manifest)")
            n_skip += 1
            append_log(log_path, {
                "time": _utc_now(), "batch_id": batch_id, "file": fp.name,
                "file_id": fid, "status": "skip_duplicate",
            })
            continue

        try:
            X, yt, y3, sid, stats = preprocess_session(fp)
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {fp.name}: {e}")
            append_log(log_path, {
                "time": _utc_now(), "batch_id": batch_id, "file": fp.name,
                "file_id": fid, "status": "fail", "error": str(e),
            })
            continue

        if len(yt) == 0:
            n_empty += 1
            # 仍写入清单，避免空会话被反复重试
            manifest["files"][fid] = {
                **fp_info,
                "n_trials": 0,
                "processed_at": _utc_now(),
                "batch_id": batch_id,
                "stats": stats,
            }
            save_manifest(manifest_path, manifest)
            append_log(log_path, {
                "time": _utc_now(), "batch_id": batch_id, "file": fp.name,
                "file_id": fid, "status": "ok_empty", "stats": stats,
            })
            print(f"  empty {fp.name}", stats)
            if delete_raw_after_ok:
                fp.unlink(missing_ok=True)
            continue

        # 先缓存本批，最后统一 concat 落盘（减少反复读写大 npy）
        new_Xs.append(X)
        new_yts.append(yt)
        new_y3s.append(y3)
        new_sids.append(sid)

        manifest["files"][fid] = {
            **fp_info,
            "n_trials": int(len(yt)),
            "n_task": int(np.sum(yt == 1)),
            "n_rest": int(np.sum(yt == 0)),
            "subject": str(sid[0]) if len(sid) else "",
            "processed_at": _utc_now(),
            "batch_id": batch_id,
            "stats": stats,
        }
        # 每成功一个文件就存清单，中断也不丢「已处理」记录
        save_manifest(manifest_path, manifest)
        append_log(log_path, {
            "time": _utc_now(), "batch_id": batch_id, "file": fp.name,
            "file_id": fid, "status": "ok", "n_trials": int(len(yt)),
            "stats": stats,
        })
        n_ok += 1
        print(f"  ok {fp.name} n={len(yt)}", stats)

        if delete_raw_after_ok:
            fp.unlink(missing_ok=True)
            print(f"    deleted raw {fp.name}")

    if not new_Xs:
        print(f"本批无新样本可追加。skip={n_skip} fail={n_fail} empty={n_empty}")
        return

    new = {
        "X": np.concatenate(new_Xs, axis=0),
        "y_task": np.concatenate(new_yts, axis=0),
        "y_three": np.concatenate(new_y3s, axis=0),
        "subjects": np.concatenate(new_sids, axis=0),
    }
    sanity_check_outputs(new["X"], new["y_task"], new["y_three"])

    old = load_existing_full(out_dir)
    merged = concat_or_new(old, new)
    save_full(out_dir, merged)
    print(
        f"全量已保存: N={len(merged['X'])}"
        f"（本批 +{len(new['X'])}；此前 {0 if old is None else len(old['X'])}）"
    )

    # ---- train/val ----
    if rebuild_split:
        parts = split_all_trials(
            merged["X"], merged["y_task"], merged["y_three"],
            val_ratio=val_ratio, seed=seed, subjects=merged["subjects"],
        )
        for split in ("train", "val"):
            Xs_, yt_, y3_, sid_ = parts[split]
            np.save(_npy_path(out_dir, "X", split), Xs_)
            np.save(_npy_path(out_dir, "y_task", split), yt_)
            np.save(_npy_path(out_dir, "y_three", split), y3_)
            np.save(_npy_path(out_dir, "subjects", split), sid_)
        print("已按全量重建 train/val")
    else:
        parts = split_all_trials(
            new["X"], new["y_task"], new["y_three"],
            val_ratio=val_ratio, seed=seed, subjects=new["subjects"],
        )
        for split in ("train", "val"):
            Xs_, yt_, y3_, sid_ = parts[split]
            append_split_arrays(out_dir, split, Xs_, yt_, y3_, sid_)
        print("已将本批新样本追加到 train/val（未打乱旧样本）")

    append_log(log_path, {
        "time": _utc_now(),
        "batch_id": batch_id,
        "status": "batch_done",
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "n_empty": n_empty,
        "n_full": int(len(merged["X"])),
        "n_batch_new": int(len(new["X"])),
    })
    print(f"done ok={n_ok} skip={n_skip} fail={n_fail} empty={n_empty} → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stieger 增量预处理批处理")
    # 路径相对 batch.py 所在：.../src/datasets/stieger/batch.py
    # parents[3]=preprocess_lab，parents[5]=仓库根 MI/
    _PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
    _REPO_ROOT = Path(__file__).resolve().parents[5]
    parser.add_argument(
        "--glob",
        default=str(_REPO_ROOT / "DATA" / "stieger" / "S*_Session_*.mat"),
        help="本批原始 mat 通配符",
    )
    parser.add_argument(
        "--out",
        default=str(_PREPROCESS_ROOT / "out" / "stieger_2s"),
    )
    parser.add_argument("--delete-raw", action="store_true", help="成功后删除原始 mat")
    parser.add_argument("--rebuild-split", action="store_true", help="用全量重划 train/val")
    args = parser.parse_args()

    run_incremental_batch(
        data_glob=args.glob,
        out_dir=Path(args.out),
        delete_raw_after_ok=args.delete_raw,
        rebuild_split=args.rebuild_split,
    )


if __name__ == "__main__":
    main()
```

#### 5.8.5 清单示例（`processed_manifest.json`）

```json
{
  "version": 1,
  "files": {
    "S1_Session_1": {
      "size": 123456789,
      "sha1_sample": "abc...",
      "mtime": 1720000000,
      "n_trials": 120,
      "n_task": 80,
      "n_rest": 40,
      "subject": "S1",
      "processed_at": "2026-07-25T04:00:00Z",
      "batch_id": "2026-07-25T04:00:00Z",
      "stats": {"n_raw": 450, "n_keep": 120, "n_drop_short": 30}
    }
  }
}
```

#### 5.8.6 若某文件要强制重跑

从 `processed_manifest.json` 的 `files` 里删掉对应键，把 `.mat` 再放回 `DATA/stieger/`，重新跑批处理即可。  
**不要**直接改 `stieger_X.npy` 去重——清单才是防重复入口；全量 npy 本身是追加日志，重跑前若需去重应备份后重建。

#### 5.8.7 磁盘与安全建议

| 建议 | 说明 |
|------|------|
| 先关删除 | 前几批保持 `delete_raw_after_ok: false`，确认清单与 N 增长正确 |
| 先追加全量 | 正式评估用 `stieger_*.npy`；`train_/val_` 可有可无 |
| 中断可续跑 | 每文件成功即写清单；重跑会 skip 已完成文件 |
| 备份清单 | 删原始前可把 `processed_manifest.json` 拷一份到网盘 |

---

### 5.9 注册 loader（`src/datasets/registry.py`）

**路径：** `code/preprocess_lab/src/datasets/registry.py`

```python
from src.datasets.bci2a.load_mat import load_bci2a_mat
from src.datasets.stieger.load_mat import load_stieger_mat

LOADERS = {
    "load_bci2a_mat": load_bci2a_mat,
    "load_stieger_mat": load_stieger_mat,
}
```

> 说明：2a 的 `batch.py` 走 `ContinuousEEG` + `preprocess_run`；Stieger 的 `batch.py` 走 `StiegerTrial` + `preprocess_session`。**两套 batch 并列，不要互相塞。**

---

## 6. 如何运行

在 `code/preprocess_lab` 下：

```bash
# 1) 单会话冒烟
python -m src.datasets.stieger.pipeline

# 2) 本批增量（DATA/stieger 里当前有的 mat）
python -m src.datasets.stieger.batch

# 3) 确认无误后，成功即删原始（腾盘再下载下一批）
python -m src.datasets.stieger.batch --delete-raw
```

分批推荐顺序：

1. 下载一小批 `.mat` → `DATA/stieger/`  
2. 跑增量批处理 → 追加 `out/stieger_2s/stieger_*.npy`，更新 `processed_manifest.json`  
3. 核对：新文件在清单中、`N` 增加、无 `fail`  
4. 删除本批原始（或 `--delete-raw`）→ 下载下一批 → 回到步骤 2  
5. 全部完成后，用全量 `stieger_*.npy` + `subjects` 做被试五折；权重目录与 BCI2a 分开、从零训练  

---

## 7. 验收清单

- [ ] `X.shape[1:] == (1, 8, 500)`，dtype float32  
- [ ] `y_three==0` 当且仅当 `y_task==0`  
- [ ] 任务样本只有左=1 / 右=2；无 3（双手）泄漏  
- [ ] 通道序为 `Cz,C3,C4,CP3,FC4,FC3,CP4,CPz`  
- [ ] 随机抽 5 条：每通道 mean≈0、std≈1  
- [ ] `subjects` 非空，五折可按人切  
- [ ] 2a 与 Stieger 分别位于 `src/datasets/bci2a/`、`src/datasets/stieger/`，标签互不混用  
- [ ] 共用步骤仅从 `src/common/steps/` 导入  
- [ ] 同一文件跑两遍：第二次全部 `skip`，`N` 不变  
- [ ] 第二批新文件：`N` 只增加本批试次数，旧清单条目仍在  
- [ ] `batch_log.jsonl` 每批有 `batch_done`  

---

## 8. 后续扩展（当前范式已含 UD/2D 左右与静息）

| 项 | 说明 |
|----|------|
| ITI 人造静息 | 与原生 down 分开报告；默认仍关闭 |
| 噪声通道插值 | 用 `noisechan` + MNE interpolate |
| 全 598 会话过夜 | 建议先子集估时再全量 |
| 与 2a 对比实验 | 各自 `out/*` 从零训 EEGNet；禁止加载对方 `best_*.pt` |
| 仅 LR 消融 | 将 `use_tasks` 改回 `(1,)` 可对比无静息时的影响 |

---

## 9. 文件落点速查表（按目标结构）

| 文件 | 动作 |
|------|------|
| `src/common/eeg_types.py` 等 | 由旧 `src/` 顶层 / `steps/` **迁入** |
| `src/datasets/bci2a/load_mat.py` | 由 `io/load_bci2a_mat.py` **迁入并改名** |
| `src/datasets/bci2a/labels.py` | 由 `steps/harmonize_labels.py` **迁入** |
| `src/datasets/bci2a/pipeline.py` | 由 `src/pipeline.py` **迁入** |
| `src/datasets/bci2a/batch.py` | 由 `src/pipline_batch.py` **迁入并改名** |
| `src/datasets/stieger/__init__.py` | 新建或由 `io/stieger/` **迁入** |
| `src/datasets/stieger/labels.py` | 新建 / 迁入 |
| `src/datasets/stieger/paradigm.py` | 新建 / 迁入 |
| `src/datasets/stieger/windows.py` | 新建 / 迁入 |
| `src/datasets/stieger/load_mat.py` | 新建 / 迁入 |
| `src/datasets/stieger/pipeline.py` | 由 `pipeline_stieger.py` **迁入并改名** |
| `src/datasets/stieger/batch.py` | **已实现**（§5.8；默认 `out/stieger_2s`） |
| `src/datasets/registry.py` | 由 `io/registry.py` **迁入** |
| `config/bci2a.yaml` / `config/stieger.yaml` | 配置并列 |
| `out/stieger/processed_manifest.json` | 运行时生成 |
| `out/stieger/batch_log.jsonl` | 运行时生成 |

> **本文阶段**：目录已按 §3 迁移；`stieger/batch.py` 已按 §5.8 落地。

---

## 10. 关键引用

- 数据描述论文：https://doi.org/10.1038/s41597-021-00883-1  
- Figshare：https://doi.org/10.6084/m9.figshare.13123148  
- 本仓库适配总表：`资料/数据集说明/数据集适配分析_扩展推荐.md`  
- 多库结构审阅稿：`资料/预处理/统一多数据集预处理框架_结构示例_审阅稿.md`（目录已与本节对齐）  
- BCI2a 目标流水线：`code/preprocess_lab/src/datasets/bci2a/pipeline.py`
