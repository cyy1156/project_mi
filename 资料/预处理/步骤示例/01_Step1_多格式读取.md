# Step 1：多格式数据读取

## 目标

为每种格式写 `load_xxx()`，统一返回 `ContinuousEEG`（或 `list[ContinuousEEG]`）。  
**先只做 BCI IV 2a 的 `.mat`**，其它格式后做。

建议数据路径（若你工程里有）：`MI_model/DATA/archive/A01T.mat` ~ `A09T.mat`

---

## 1.1 BCI IV 2a — `.mat`（优先完成）

文件建议：`preprocess_lab/src/io/load_bci2a_mat.py`

### 通道名（22 EEG，与常见 mat 一致）

```python
EEG22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
```

### 参考示例

```python
from pathlib import Path
import numpy as np
import scipy.io
# from your types import ContinuousEEG


def load_bci2a_mat(mat_path: Path) -> list[ContinuousEEG]:
    """读取 BCI IV 2a Training .mat，每个带标签 run → 一条 ContinuousEEG。"""
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

        events = np.column_stack([trial.astype(int), y])

        out.append(
            ContinuousEEG(
                subject=subject,
                session=f"run{run_id}",
                x=x[:, :22],          # 丢掉 EOG；真正 8 通道筛选在 Step3
                fs=fs,
                ch_names=list(EEG22),
                events=events,
                labels=y,
                artifacts=artifacts,
            )
        )
    return out
```

### 你本地应立刻打印

```python
runs = load_bci2a_mat(Path(r".../A01T.mat"))
for eeg in runs:
    sanity_check(eeg)
    print("y unique:", np.unique(eeg.labels))
```

期望大致：

- `fs == 250`
- `x.shape[1] == 22`
- `events` 行数与 `labels` / `artifacts` 一致（有标签的 run）
- `y` 取值在 `{1,2,3,4}`

---

## 1.2 `.gdf`（第二阶段）

```python
import mne
from pathlib import Path


def load_gdf(gdf_path: Path) -> ContinuousEEG:
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
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

注意：不同数据集的 `event_id` 映射不同，**必须对照官方说明单独写表**，不要抄 2a 的 1/2/3/4。

---

## 1.3 OpenBCI CSV（第二阶段）

```python
import pandas as pd
import numpy as np
from pathlib import Path


def load_openbci_csv(csv_path: Path, fs: float = 250.0) -> ContinuousEEG:
    df = pd.read_csv(csv_path)
    ch_names = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
    x = df[ch_names].to_numpy(dtype=np.float64)

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

列名 / event 码按你自己的采集协议改。

---

## 常见坑

- 校准 run 的 `trial` 为空，要跳过，否则后面全崩。
- `squeeze_me=True` 后单元素可能变成标量，用 `np.atleast_1d`。
- 先保留 22 EEG；**不要**在 Step1 就偷偷只留 8 通道（筛选是 Step3 的职责，便于对照文档）。

## 验收清单

- [ ] `load_bci2a_mat(A01T.mat)` 返回多条 `ContinuousEEG`
- [ ] 每条通过 `sanity_check`
- [ ] `x` 为 `(T, 22)`，`fs=250`
- [ ] 有标签 run 的 `events[:,1]` 与原始 `y` 一致

## 提交检查时附上

`load_bci2a_mat` 完整代码 + A01 每个 run 的 `shape / fs / events` 打印。
