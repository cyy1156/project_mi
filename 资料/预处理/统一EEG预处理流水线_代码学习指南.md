# 统一 EEG 预处理流水线 —— 代码学习指南

> 依据文档：  
> 1. `统一EEG数据预处理流水线（Unified EEG Preprocessing Pipeline）——跨数据集通用标准.docx`  
> 2. `数据集说明.docx`  
> 3. `面向少样本个体适配的运动想象脑机接口系统设计.docx`（§3.1.2 双标签）  
>  
> 本文目标：用**可对照的代码示例**帮你理解「如何按统一标准处理多格式原始 EEG」，不是可直接跑全量实验的工程代码。  
> 学完后，你应能自己把示例拼成 `loader → preprocess → save` 流水线。

---

## 0. 先记住最终输出长什么样

所有数据集走完本流水线后，必须落到同一规格：

| 项目 | 统一标准 |
|------|----------|
| 通道 | **8** 个，顺序固定（见下） |
| 采样率 | **250 Hz** |
| 时间长度 | **4 s → 1000 点** |
| 标签 | **双标签**（同一样本两套）：见下表 |
| 张量 | `(N, 1, 8, 1000)`（样本只存一份，不按分类头复制） |
| 幅值 | **单试次、单通道 Z-score** |

双标签映射（项目计划 §3.1.2）：

| 样本 | `label_task`（静息/任务） | `label_three`（空闲/左/右） |
|------|---------------------------|------------------------------|
| 静息 | 0 | 0 |
| 左手 | 1 | 1 |
| 右手 | 1 | 2 |

固定通道顺序（输入通道轴顺序，**禁止**按字母或原文件顺序重排）：

```text
索引:  0    1    2    3     4     5     6     7
名称: Cz   C3   C4   CP3   FC4   FC3   CP4   CPz
```

完整流程：

```text
Raw EEG
  → Step1 多格式读取
  → Step2 标签同质化（只留左/右；造静息；丢脚/舌；写出双标签）
  → Step3 选 8 通道
  → Step4 CAR
  → Step5 Notch 50Hz + Bandpass 8–30Hz
  → Step6 Epoch（Cue 前 -0.5s ~ Cue 后 4.0s）
  → Step7 基线校正 [-0.5, 0]
  → Step8 分类窗 Cue 后 0~4s（等价绝对时间 t=2~6s）
  → Step9 重采样到 250Hz / 1000 点
  → Step10 单试次 Z-score
  → Step11 张量规整 (N,1,8,1000)
  → Step12 按被试划分 train/val/test
```

---

## 1. 不同数据集：格式差异一览

来自《数据集说明》，写代码前先搞清「每种原始文件长什么样」。

| 数据集 | 常见格式 | 采样率 | 通道规模 | 类别要点 |
|--------|----------|--------|----------|----------|
| BCI IV 2a | `.mat` / `.gdf` | 250 Hz | 22 EEG + 3 EOG | 原 4 类；本流水线只留左/右，静息另造 |
| BCI IV 2b | `.gdf` | 250 Hz | 3（C3/Cz/C4） | 左/右；通道不足 8 时本标准要求**不补零**，需单独策略 |
| BCI IV 1 | `.mat` | 1000→本地常 100 | 59 | 两动作 + 空闲 |
| OpenBMI / Lee2019 | `.mat` / BIDS | 1000 Hz | 62 | 左/右；需降采到 250 |
| OpenBCI 自采 | `.csv` | 常 250 Hz | 自研 8 通道 | 目标拓扑应已对齐 |
| Stieger 2021 | BIDS / `.bdf` | 1000 Hz | 60 | 左/右/双手/静息 |

**设计原则**：每种格式写一个 `load_xxx()`，返回**同一中间结构**，后面步骤不再关心文件后缀。

推荐中间结构（示例）：

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ContinuousEEG:
    """连续原始信号（尚未滤波、尚未切 epoch）。"""
    subject: str
    session: str
    x: np.ndarray          # shape: (n_times, n_channels)，单位建议统一为 μV
    fs: float              # 原始采样率
    ch_names: list[str]    # 与 x 的列一一对应
    events: np.ndarray     # shape: (n_events, 2) → [sample_index, event_id]
    labels: np.ndarray | None  # 若事件自带类别，可同步给出
    artifacts: np.ndarray | None  # 试次级伪迹标记，1=坏
```

---

## 2. Step 1：多格式数据读取（示例）

### 2.1 BCI IV 2a — `.mat`（你当前桌面工程里已有同类写法）

你现有工程可参考：

- `MI_model/src/data/loader.py`（读取 `A0xT.mat`）
- `MI_model/DATA/BCI_IV_2a_数据结构说明.md`

学习示例（精简版，突出字段）：

```python
from pathlib import Path
import numpy as np
import scipy.io

# BCI IV 2a 官方 22 EEG 通道名（顺序与常见 mat 一致）
EEG22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

def load_bci2a_mat(mat_path: Path) -> list[ContinuousEEG]:
    """读取 BCI IV 2a Training .mat，每个带标签 run 变成一条 ContinuousEEG。"""
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    subject = mat_path.stem[:3]  # A01
    out: list[ContinuousEEG] = []

    for run_id, run in enumerate(mat["data"]):
        trial = np.atleast_1d(run.trial) if run.trial is not None else np.array([])
        if trial.size == 0:
            continue  # 校准 run：无 trial，跳过

        x = np.asarray(run.X, dtype=np.float64)  # (n_times, 25) = 22EEG+3EOG
        y = np.atleast_1d(run.y).astype(int)     # 1..4
        artifacts = np.atleast_1d(run.artifacts).astype(int)
        fs = float(run.fs)

        # events: cue 采样点 + 原始类别
        events = np.column_stack([trial.astype(int), y])

        out.append(
            ContinuousEEG(
                subject=subject,
                session=f"run{run_id}",
                x=x[:, :22],          # 先丢掉 EOG；通道筛选在 Step3
                fs=fs,
                ch_names=EEG22,
                events=events,
                labels=y,
                artifacts=artifacts,
            )
        )
    return out
```

### 2.2 BCI IV 2a / 2b — `.gdf`（MNE）

```python
import mne

def load_gdf(gdf_path: Path) -> ContinuousEEG:
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    # 事件：不同数据集 event_id 映射不同，需对照官方说明单独写表
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return ContinuousEEG(
        subject=gdf_path.stem[:3],
        session=gdf_path.stem,
        x=raw.get_data().T,          # → (n_times, n_ch)
        fs=float(raw.info["sfreq"]),
        ch_names=list(raw.ch_names),
        events=events[:, [0, 2]],    # sample, event_code
        labels=None,
        artifacts=None,
    )
```

### 2.3 OpenBCI 自采 — `.csv`

自采 CSV 常见列：`timestamp, Cz, C3, ...` 或 `ch1..ch8` + 单独 event 列。下面是**示意**：

```python
import pandas as pd

def load_openbci_csv(csv_path: Path, fs: float = 250.0) -> ContinuousEEG:
    df = pd.read_csv(csv_path)
    # 假设已经按统一 8 通道命名导出
    ch_names = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
    x = df[ch_names].to_numpy(dtype=np.float64)

    # 假设有 event 列：0=无，1=左，2=右，3=开始静息标记等（按你自己协议改）
    if "event" in df.columns:
        idx = np.where(df["event"].to_numpy() != 0)[0]
        codes = df["event"].to_numpy()[idx]
        events = np.column_stack([idx, codes])
    else:
        events = np.zeros((0, 2), dtype=int)

    return ContinuousEEG(
        subject=csv_path.stem,
        session="session0",
        x=x,
        fs=fs,
        ch_names=ch_names,
        events=events,
        labels=None,
        artifacts=None,
    )
```

### 2.4 OpenBMI 等 1000 Hz `.mat`

结构因版本而异，原则相同：**取出连续 EEG + 采样率 + 通道名 + cue 时间**，再进入同质化。读通后立刻检查：

```python
def sanity_check(eeg: ContinuousEEG) -> None:
    assert eeg.x.ndim == 2
    assert eeg.x.shape[1] == len(eeg.ch_names)
    assert eeg.fs > 0
    print(eeg.subject, eeg.x.shape, eeg.fs, "Hz", "events=", len(eeg.events))
```

---

## 3. Step 2：数据集同质化（双标签与样本规则）

与旧版四分类、以及「单标签左=0/右=1/静=2」均不同：本项目采用**双标签**，供共享主干上的两个分类头监督（见项目计划 §3.1.2）。

| 原始情况 | `label_task` | `label_three` |
|----------|--------------|---------------|
| 左手 | **1**（任务） | **1** |
| 右手 | **1**（任务） | **2** |
| 静息 | **0**（静息） | **0**（空闲） |
| 双脚 / 舌头（BCI 2a） | **直接丢弃**，不得标成静息 | — |
| 伪迹 trial | 丢弃 | — |

BCI IV 2a 时间约定：

- Cue 在试次绝对时间 **t=2.0 s** 出现  
- 左/右分类窗：**t=2~6 s** ≡ **Cue 后 0~4 s**

学习示例：只保留左/右事件，并写成双标签。

```python
# BCI IV 2a 原始 y: 1=左, 2=右, 3=脚, 4=舌
THREE_MAP = {1: 1, 2: 2}  # → label_three；任务样本 label_task 恒为 1

def filter_left_right_events(
    events: np.ndarray,
    artifacts: np.ndarray | None,
) -> np.ndarray:
    """返回 shape (n_keep, 4): [cue_sample, label_task, label_three, trial_index]。"""
    kept = []
    for i, (samp, y) in enumerate(events):
        if artifacts is not None and artifacts[i] == 1:
            continue
        if int(y) not in THREE_MAP:
            continue  # 脚/舌直接丢
        kept.append([int(samp), 1, THREE_MAP[int(y)], i])
    return np.asarray(kept, dtype=int) if kept else np.zeros((0, 4), dtype=int)
```

静息窗构造（逻辑示意，实现时要仔细处理边界；标签固定为 `task=0, three=0`）：

```python
def extract_rest_cues(
    cue_samples: np.ndarray,
    fs: float,
    n_times: int,
    rest_sec: float = 4.0,
) -> list[int]:
    """优先取「下一次 Cue 前 4s」作为静息起点；不足/越界则丢弃。"""
    rest_len = int(rest_sec * fs)
    starts = []
    cues = np.sort(cue_samples.astype(int))
    for i in range(len(cues) - 1):
        # 下一 Cue 前 rest_len 个点
        start = cues[i + 1] - rest_len
        end = cues[i + 1]
        # 还要保证不与「上一任务窗 Cue~Cue+4s」重叠 —— 此处仅示意
        prev_task_end = cues[i] + int(4.0 * fs)
        if start < 0 or end > n_times:
            continue
        if start < prev_task_end:
            continue
        starts.append(start)
    return starts
```

> 论文/文档要求：静息样本数量尽量与左/右手按受试者、会话平衡。

---

## 4. Step 3：通道统一筛选

```python
TARGET_CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]

def select_channels(x: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """
    x: (n_times, n_ch)
    缺失通道：本标准规定不补零、不插值 → 直接报错，便于发现数据问题。
    """
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    missing = [c for c in TARGET_CHANS if c not in name_to_idx]
    if missing:
        raise KeyError(f"缺少目标通道: {missing}；可用: {ch_names}")
    idx = [name_to_idx[c] for c in TARGET_CHANS]
    return x[:, idx]  # (n_times, 8)，顺序已固定
```

> 注意：BCI IV 2b 只有 C3/Cz/C4，无法满足 8 通道标准；要么排除该数据集，要么另定「低导联」实验分支，**不要偷偷补零**。

---

## 5. Step 4–5：CAR + 双层滤波

文档要求：**CAR 在滤波之前**。

```python
import mne

def car_reference(x: np.ndarray) -> np.ndarray:
    """x: (n_times, n_ch) → 每时刻减去全通道均值。"""
    return x - x.mean(axis=1, keepdims=True)

def notch_and_bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    """
    x: (n_times, n_ch)
    Notch 50 Hz + Bandpass 8–30 Hz
    mne.filter 期望 (n_ch, n_times)，所以要转置。
    """
    data = x.T  # (n_ch, n_times)
    data = mne.filter.notch_filter(data, Fs=fs, freqs=50.0, verbose=False)
    data = mne.filter.filter_data(
        data, sfreq=fs, l_freq=8.0, h_freq=30.0, verbose=False
    )
    return data.T
```

---

## 6. Step 6–8：Epoch → 基线校正 → 分类窗

左/右手：

- Epoch：Cue 前 **-0.5 s** ~ Cue 后 **4.0 s**  
- 基线：`[-0.5, 0]`  
- 分类输入：Cue 后 **0~4 s**

```python
def slice_epoch(x: np.ndarray, cue: int, fs: float) -> np.ndarray | None:
    """返回 (n_times_epoch, n_ch)，含基线段。"""
    t0 = cue + int(-0.5 * fs)
    t1 = cue + int(4.0 * fs)
    if t0 < 0 or t1 > x.shape[0]:
        return None
    return x[t0:t1, :]

def baseline_correct(epoch: np.ndarray, fs: float) -> np.ndarray:
    """epoch 从 -0.5s 开始；用前 0.5s 均值归零。"""
    b1 = int(0.5 * fs)
    base = epoch[:b1, :].mean(axis=0, keepdims=True)
    return epoch - base

def classification_window(epoch: np.ndarray, fs: float) -> np.ndarray:
    """去掉基线段，只留 Cue 后 0~4s。"""
    c0 = int(0.5 * fs)
    return epoch[c0:, :]
```

静息：直接截 4 s；文档说可用自身前 0.5 s 做基线，但**不缩短**最终 4 s 输入——实现时常见做法是：在 4 s 窗内用开头 0.5 s 均值减全窗（学习时先按左/右同一套「先扩窗再截」也可，务必在实验记录里写清）。

---

## 7. Step 9：统一重采样到 250 Hz / 1000 点

```python
from scipy.signal import resample

def resample_to_1000(x_win: np.ndarray, fs_in: float, fs_out: float = 250.0) -> np.ndarray:
    """
    x_win: (n_times_in, 8)，应对应正好 4 秒。
    输出: (1000, 8)
    """
    n_out = int(4.0 * fs_out)  # 1000
    # 若输入已是 250Hz 且长度已是 1000，可直接返回
    if abs(fs_in - fs_out) < 1e-6 and x_win.shape[0] == n_out:
        return x_win.astype(np.float32)
    y = resample(x_win, n_out, axis=0)
    return np.asarray(y, dtype=np.float32)
```

---

## 8. Step 10–11：单试次 Z-score + 张量规整

与「训练集通道 z-score」不同：本标准是 **trial-wise**（每个试次、每个通道，在 4 s 有效窗上算均值方差）。

```python
def trial_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: (1000, 8) → 同形状，每通道独立标准化。"""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std

def to_eegnet_tensor(trials: list[np.ndarray]) -> np.ndarray:
    """
    trials: 每个元素 (1000, 8)
    输出: (N, 1, 8, 1000)
    """
    arr = np.stack(trials, axis=0)          # (N, 1000, 8)
    arr = np.transpose(arr, (0, 2, 1))      # (N, 8, 1000)
    return arr[:, None, :, :].astype(np.float32)
```

---

## 9. 串成一条「单受试者」流水线（示意）

把上面步骤串起来，便于你对照文档逐步调试：

```python
def preprocess_run(
    eeg: ContinuousEEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 X, y_task, y_three。"""
    x = select_channels(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs)

    kept = filter_left_right_events(eeg.events, eeg.artifacts)
    xs, y_task, y_three = [], [], []

    for cue, lab_task, lab_three, _ in kept:
        ep = slice_epoch(x, int(cue), eeg.fs)
        if ep is None:
            continue
        ep = baseline_correct(ep, eeg.fs)
        win = classification_window(ep, eeg.fs)
        win = resample_to_1000(win, fs_in=eeg.fs, fs_out=250.0)
        if win.shape != (1000, 8):
            continue
        win = trial_zscore(win)
        xs.append(win)
        y_task.append(lab_task)
        y_three.append(lab_three)

    # TODO: 追加静息 → y_task=0, y_three=0，并做数量平衡

    if not xs:
        empty_x = np.zeros((0, 1, 8, 1000), np.float32)
        empty_y = np.zeros((0,), np.int64)
        return empty_x, empty_y, empty_y.copy()

    X = to_eegnet_tensor(xs)
    return (
        X,
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
    )
```

保存示例：

```python
np.save("A01_X.npy", X)              # (N,1,8,1000)
np.save("A01_y_task.npy", y_task)    # (N,)  0=静息, 1=任务
np.save("A01_y_three.npy", y_three)  # (N,)  0=空闲, 1=左手, 2=右手
```

---

## 10. Step 12：按被试划分（示意）

文档强调：**按被试划分**，不要把同一人的试次打散到 train/test 后假装「跨被试」。

```python
def split_by_subject(
    X: np.ndarray,
    y_task: np.ndarray,
    y_three: np.ndarray,
    subjects: list[str],
    test_subjects: set[str],
    val_subjects: set[str],
) -> dict:
    subj = np.asarray(subjects)
    masks = {
        "train": ~np.isin(subj, list(test_subjects | val_subjects)),
        "val": np.isin(subj, list(val_subjects)),
        "test": np.isin(subj, list(test_subjects)),
    }
    return {k: (X[m], y_task[m], y_three[m]) for k, m in masks.items()}
```

---

## 11. CAR / 基线 / Z-score：写代码时别混

| 操作 | 改什么 | 用哪段数据 | 代码位置 |
|------|--------|------------|----------|
| CAR | 空间公共噪声 | 连续信号整段 | 滤波前 |
| 基线校正 | 直流偏移 | Epoch 的 `[-0.5,0]` | 切 epoch 后 |
| Trial Z-score | 幅值分布 | 分类窗 4 s | 重采样后、进模型前 |

三者都要做，顺序不能乱。

---

## 12. 和你现有 `MI_model` 工程的区别（重要）

你桌面上的 `model/MI_model` 是**另一套已跑通的四分类基线**，参数不同：

| 项目 | 现有 `MI_model` | 本文统一流水线 |
|------|-----------------|----------------|
| 通道 | 22 | **8** |
| 采样率 / 点数 | 125 Hz / 500 | **250 Hz / 1000** |
| 类别 | 4 类（含脚、舌） | **双标签**：task∈{0,1} + three∈{0,1,2}（静/左/右） |
| 标准化 | 训练集通道 z-score | **单试次 z-score** |
| 参考代码 | `src/data/preprocess.py` | 本文示例（需新建） |

学习建议：

1. 先读懂 `MI_model` 的 `loader.py` + `preprocess.py`（流程骨架很像）  
2. 再按本文标准**改参数与标签逻辑**，不要直接把旧 `epochs_npy` 当成本流水线输出  
3. 原始 `.mat` 位置仍是：`MI_model/DATA/archive/A01T.mat` ~ `A09T.mat`

---

## 13. 建议你本地落地的文件结构（自学用）

等你开始正式写工程时，可以按模块拆文件（仍是建议，不是已生成代码）：

```text
preprocess_lab/
  README.md                      # 可放本文副本
  configs/pipeline.yaml          # 8通道、250Hz、8-30Hz 等
  src/
    io/
      load_bci2a_mat.py
      load_gdf.py
      load_openbci_csv.py
    steps/
      harmonize_labels.py
      select_channels.py
      filter_car.py
      epoch_baseline.py
      resample_zscore.py
    pipeline.py                  # 串步骤
  scripts/
    run_one_subject.py           # 先跑 A01 验证 shape
    run_all.py
  out/
    A01_X.npy
    A01_y_task.npy
    A01_y_three.npy
```

验收一条（A01）：

```text
X.shape == (n, 1, 8, 1000)
y_task ∈ {0,1}；y_three ∈ {0,1,2}
映射一致：静息(0,0) / 左手(1,1) / 右手(1,2)
无 NaN；每通道 std≈1（trial-wise 后）
左/右数量接近；静息按规则平衡
```

---

## 14. 推荐学习顺序

1. 只用 `A01T.mat`，写通 `load_bci2a_mat`，打印 `x.shape / fs / events`  
2. 做通道筛选，确认输出永远是 `(T, 8)` 且顺序正确  
3. 加 CAR + 滤波，画一条 C3 滤波前后波形  
4. 只切左手/右手 epoch，检查每段长度  
5. 基线 → 分类窗 → resample → z-score → 堆成 `(N,1,8,1000)`  
6. 再实现静息窗与多数据格式 loader  

---

## 15. 依赖（学习环境）

```text
numpy
scipy
mne
pandas          # CSV / 表格
pyyaml          # 可选，存配置
```

---

*文档性质：学习指南 + 代码示例。实现全量跨数据集工程时，请把事件码表、静息平衡、边界检查写完整，并用单受试者可视化验收后再跑全员。*
