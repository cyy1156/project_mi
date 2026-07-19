# Step 6–8：Epoch → 基线校正 → 分类窗

## 目标

对**左/右手**试次：

| 步骤 | 时间定义 |
|------|----------|
| Epoch | Cue 前 **-0.5 s** ~ Cue 后 **4.0 s**（共 4.5 s） |
| 基线 | Epoch 内相对时间 `[-0.5, 0]`（开头 0.5 s） |
| 分类窗 | Cue 后 **0~4 s**（丢掉基线段） |

输入 `x` 应已是：选通道 → CAR → 滤波后的连续信号 `(T, 8)`。

文件建议：`preprocess_lab/src/steps/epoch_baseline.py`

## 参考示例

```python
import numpy as np


def slice_epoch(x: np.ndarray, cue: int, fs: float) -> np.ndarray | None:
    """返回 (n_times_epoch, n_ch)，含基线段；越界返回 None。"""
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

串起来：

```python
def epoch_to_class_window(x: np.ndarray, cue: int, fs: float) -> np.ndarray | None:
    ep = slice_epoch(x, cue, fs)
    if ep is None:
        return None
    ep = baseline_correct(ep, fs)
    return classification_window(ep, fs)
```

### 长度检查（250 Hz 时）

- Epoch：`0.5*250 + 4*250 = 1125` 点（注意：右端开区间写法时，`t1 - t0` 应等于 `int(4.5*fs)`）
- 分类窗：`4*250 = 1000` 点

请用你自己的切片约定验证：

```python
win = epoch_to_class_window(x, int(kept[0, 0]), eeg.fs)
print(win.shape)  # 期望约 (1000, 8) @ 250Hz
```

若 `int` 截断导致差 1 个点，后面 Step9 仍会 resample 到 1000；但 250 Hz 源数据应尽量直接得到 1000。

---

## 静息窗说明

静息直接截 4 s；文档允许用自身前 0.5 s 做基线，但**不缩短**最终 4 s 输入。  
切出的静息样本标签固定为：`label_task=0`，`label_three=0`（见 Step2 双标签规则）。

两种常见实现（选一种并写进实验记录）：

**A. 与左/右同一套「先扩窗再截」**（推荐学习时统一）

- 静息起点 `s` 视为「假 Cue」
- 仍用 `slice_epoch(x, s + int(0.5*fs), fs)` 等你需要自己推算对齐方式  
  更直观的做法见 B。

**B. 直接 4 s + 窗内基线**

```python
def rest_window_with_baseline(x: np.ndarray, start: int, fs: float) -> np.ndarray | None:
    """截 [start, start+4s)，用开头 0.5s 均值减全窗，长度仍为 4s。"""
    n = int(4.0 * fs)
    if start < 0 or start + n > x.shape[0]:
        return None
    win = x[start:start + n, :].copy()
    b = int(0.5 * fs)
    win = win - win[:b, :].mean(axis=0, keepdims=True)
    return win
```

学习阶段：**左/右先跑通**，静息可先 TODO，但接口预留好。

## 常见坑

- 基线算完后误把基线段留在最终输入里（多出 0.5 s）。
- `t1` 用了闭区间导致长度 +1。
- 在**未滤波**的原始 `x` 上切窗。
- Cue 索引用错成秒数。

## 验收清单

- [ ] 越界 cue 返回 `None` 并被跳过
- [ ] 基线后，基线段均值接近 0（每通道）
- [ ] 分类窗长度在 250 Hz 下为 1000（或与 `int(4*fs)` 一致）
- [ ] 形状始终 `(n_times, 8)`

## 提交检查时附上

三个函数（或封装）+ 若干试次的 `ep.shape` / `win.shape` 打印。
