# Step 6–8：切窗 + 基线校正（BCI2a 现行）

## 目标

对**左/右手**试次（BCI IV 2a 正式设定）：

| 步骤 | 时间定义 |
|------|----------|
| 分类窗 | **Cue 后 2~4 s**（共 2 s） |
| 基线 | 分类窗起点前 **0.5 s**（Cue+1.5~2.0 s）均值减全窗 |

对**静息**：

| 步骤 | 时间定义 |
|------|----------|
| 分类窗 | **下一 Cue 前 2 s** |
| 基线 | 窗内开头 0.5 s 均值减全窗（不缩短最终 2 s） |

输入 `x` 应已是：选通道 → CAR → 滤波后的连续信号 `(T, 8)`。

文件：`preprocess_lab/src/common/steps/epoch_baseline.py`  
（`task_window_cue_2_to_4` / `rest_window_with_baseline`）

> 旧版「Cue 前 -0.5 ~ Cue 后 4.0 → 再截 0~4s」仍保留在同文件作兼容，**正式 2a 流水线已不用**。

## 参考示例

```python
import numpy as np


def task_window_cue_2_to_4(
    x: np.ndarray,
    cue: int,
    fs: float,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    """任务：Cue+2~4s；基线=窗起点前 baseline_sec。返回 (2*fs, n_ch)。"""
    n_win = int(round(2.0 * fs))
    n_base = int(round(baseline_sec * fs))
    t0 = cue + int(round(2.0 * fs))
    t1 = t0 + n_win
    base_start = t0 - n_base
    if base_start < 0 or t1 > x.shape[0]:
        return None
    base = x[base_start:t0].mean(axis=0, keepdims=True)
    win = x[t0:t1] - base
    if win.shape[0] != n_win:
        return None
    return win.astype(np.float64)


def rest_window_with_baseline(
    x: np.ndarray,
    start: int,
    fs: float,
    win_sec: float = 2.0,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    """截 [start, start+win_sec)，用开头 baseline_sec 均值减全窗。"""
    n = int(round(win_sec * fs))
    if start < 0 or start + n > x.shape[0]:
        return None
    win = x[start:start + n, :].copy()
    b = int(round(baseline_sec * fs))
    if b <= 0 or b >= n:
        return None
    win = win - win[:b, :].mean(axis=0, keepdims=True)
    return win
```

### 长度检查（250 Hz 时）

- 任务 / 静息分类窗：`2*250 = 500` 点

```python
win = task_window_cue_2_to_4(x, int(kept[0, 0]), eeg.fs)
print(win.shape)  # 期望 (500, 8) @ 250Hz
```

若 `int` 截断导致差 1 个点，后面 Step9 仍会按 `win_sec=2.0` resample 到 500；但 250 Hz 源数据应尽量直接得到 500。

---

## 时间轴示意（相对 Cue）

```text
Cue: 0s
范式 MI:     |←—————— 0 ~ 4 s ——————→|
本项目任务窗:              |← 2~4 s →|   ← 分类用
基线:                 |←0.5s→|
```

---

## 常见坑

- 仍按旧规则切「Cue 后 0~4s」却期望 500 点。
- 基线误用 Cue 前 -0.5~0（旧 epoch 方案），与现行「窗起点前 0.5s」不一致。
- 在**未滤波**的原始 `x` 上切窗。
- Cue 索引用错成秒数。

## 验收清单

- [ ] 越界 cue 返回 `None` 并被跳过
- [ ] 任务窗长度在 250 Hz 下为 500（`int(2*fs)`）
- [ ] 形状始终 `(n_times, 8)`
- [ ] 静息窗同样 500 点，标签 `(task=0, three=0)`

## 提交检查时附上

切窗函数 + 若干试次的 `win.shape` 打印。
