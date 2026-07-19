# Step 2：标签同质化（双标签：任务态 + 三分类）

> 依据项目计划 **§3.1.2 共享主干双分类头与双标签设计**：  
> 每个 EEG 样本**不复制两份**，在同一条记录上同时保存两套监督标签。

## 目标

把各数据集的原始标签，统一成**双标签**，供后续两个分类头使用：

| 标签 | 含义 | 取值 |
|------|------|------|
| `label_task` | 任务状态（分类头1：静息态 / 任务态） | `0`=静息，`1`=任务 |
| `label_three` | 最终三分类（分类头2：空闲 / 左手 / 右手） | `0`=空闲，`1`=左手，`2`=右手 |

### 双标签映射规则（必须背下来）

| 样本类型 | `label_task` | `label_three` |
|----------|--------------|---------------|
| 静息（空闲） | **0** | **0** |
| 左手想象 | **1** | **1** |
| 右手想象 | **1** | **2** |

### 样本取舍规则（与旧单标签流水线相同）

| 原始情况 | 处理 |
|----------|------|
| 左手（BCI 2a 的 `y=1`） | 保留 → `(task=1, three=1)` |
| 右手（BCI 2a 的 `y=2`） | 保留 → `(task=1, three=2)` |
| 双脚 / 舌头（BCI 2a 的 3/4） | **丢弃**（不得标成静息） |
| 伪迹 trial（`artifacts==1`） | **丢弃** |
| 静息 | 从无 Cue、无 MI、无伪迹的休息段截 4 s → `(task=0, three=0)` |

BCI IV 2a 时间约定（后续切窗要用）：

- Cue 在试次绝对时间 **t=2.0 s**
- 分类窗：**t=2~6 s** ≡ **Cue 后 0~4 s**

文件建议：`preprocess_lab/src/steps/harmonize_labels.py`

---

## 2.1 只保留左 / 右，并写成双标签

```python
import numpy as np

# BCI IV 2a 原始 y: 1=左, 2=右, 3=脚, 4=舌
# → label_three: 左=1, 右=2；任务样本的 label_task 恒为 1
THREE_MAP = {1: 1, 2: 2}


def filter_left_right_events(
    events: np.ndarray,
    artifacts: np.ndarray | None,
) -> np.ndarray:
    """
    返回 shape (n_keep, 4):
      [cue_sample, label_task, label_three, trial_index]

    - 左/右均为任务态：label_task=1
    - label_three: 左手=1，右手=2
    trial_index 便于回溯原始 trial / artifacts。
    """
    kept = []
    for i, (samp, y) in enumerate(events):
        if artifacts is not None and int(artifacts[i]) == 1:
            continue
        if int(y) not in THREE_MAP:
            continue  # 脚/舌直接丢
        label_task = 1
        label_three = THREE_MAP[int(y)]
        kept.append([int(samp), label_task, label_three, i])
    if not kept:
        return np.zeros((0, 4), dtype=int)
    return np.asarray(kept, dtype=int)
```

### 自测建议

对 A01 某一 run：

```python
kept = filter_left_right_events(eeg.events, eeg.artifacts)
print("kept:", kept.shape)  # (n, 4)
print("label_task unique:", np.unique(kept[:, 1]))   # 只应有 {1}
print("label_three unique:", np.unique(kept[:, 2]))  # 只应有 {1, 2}
print("dropped feet/tongue/artifact count:", len(eeg.events) - len(kept))

# 抽查：左手行应满足 task=1, three=1；右手行 task=1, three=2
assert np.all(kept[:, 1] == 1)
assert set(np.unique(kept[:, 2])).issubset({1, 2})
```

---

## 2.2 静息 Cue（起点）构造

逻辑：优先取「下一次 Cue 前 4 s」作为静息起点；越界或与上一任务窗重叠则丢弃。  
静息样本标签固定为：`label_task=0`，`label_three=0`。

```python
def extract_rest_cues(
    cue_samples: np.ndarray,
    fs: float,
    n_times: int,
    rest_sec: float = 4.0,
    task_sec: float = 4.0,
) -> list[int]:
    """返回静息窗起点采样索引列表；标签在流水线里写成 (0, 0)。"""
    rest_len = int(rest_sec * fs)
    task_len = int(task_sec * fs)
    starts: list[int] = []
    cues = np.sort(cue_samples.astype(int))

    for i in range(len(cues) - 1):
        start = cues[i + 1] - rest_len
        end = cues[i + 1]
        prev_task_end = cues[i] + task_len

        if start < 0 or end > n_times:
            continue
        if start < prev_task_end:  # 与上一任务窗重叠
            continue
        starts.append(int(start))
    return starts
```

辅助：把静息起点也整理成与 kept 同结构（可选）：

```python
def rest_starts_to_rows(starts: list[int]) -> np.ndarray:
    """每行: [start_sample, label_task=0, label_three=0, trial_index=-1]"""
    if not starts:
        return np.zeros((0, 4), dtype=int)
    rows = [[int(s), 0, 0, -1] for s in starts]
    return np.asarray(rows, dtype=int)
```

> 文档要求：静息数量尽量与左/右手按受试者、会话**平衡**。  
> 实现时可：先得到所有合法 rest starts，再按 `min(n_left, n_right)` 抽样或截断。平衡逻辑可先 TODO，但要在注释里写清。

### 自测建议

```python
cue_samples = kept[:, 0]
rests = extract_rest_cues(cue_samples, eeg.fs, eeg.x.shape[0])
rest_rows = rest_starts_to_rows(rests)
print("n_left:", np.sum(kept[:, 2] == 1), "n_right:", np.sum(kept[:, 2] == 2))
print("n_rest_candidates:", len(rests))
if len(rest_rows):
    assert np.all(rest_rows[:, 1] == 0) and np.all(rest_rows[:, 2] == 0)
```

---

## 和旧版单标签的区别（勿混用）

| | 旧文档（已废弃） | 当前项目计划 |
|--|------------------|--------------|
| 输出 | 单一 `y`：左=0，右=1，静=2 | **`y_task` + `y_three` 两套** |
| 左手 | `y=0` | `task=1, three=1` |
| 右手 | `y=1` | `task=1, three=2` |
| 静息 | `y=2` | `task=0, three=0` |

训练时：任务头吃 `y_task`，三分类头吃 `y_three`；样本 `X` 只保留一份。

---

## 常见坑

- 仍按旧规则把左手写成 `0`、右手写成 `1`（那是旧单标签，**不是**现在的 `label_three`）。
- 把脚/舌标成静息 → **错误**。
- 伪迹试次没丢掉。
- 静息窗与任务窗重叠仍保留。
- `THREE_MAP` 写反（1→2, 2→1）导致左右颠倒。
- 静息只写了 `label_three=0` 却忘了 `label_task=0`（或反过来不一致）。

## 验收清单

- [ ] 左/右 kept 行：`label_task` 全为 `1`，`label_three` 只含 `{1, 2}`
- [ ] 静息行：`label_task=0` 且 `label_three=0`
- [ ] 脚/舌试次不出现在 kept 里
- [ ] `artifacts==1` 的 trial 不在 kept 里
- [ ] rest 起点不越界、不与上一任务窗重叠（抽查几个索引换算成秒验证）

## 提交检查时附上

相关函数完整代码 + A01 某 run 的 `label_task` / `label_three` 统计与 rest 候选数量。
