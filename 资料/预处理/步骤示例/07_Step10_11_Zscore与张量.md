# Step 10–11：单试次 Z-score + 张量规整

## 目标

1. **Trial-wise Z-score**：每个试次、每个通道，在 4 s（1000 点）上独立标准化。  
   （不是「训练集全局通道 z-score」。）
2. 堆成 EEGNet 常用形状：`(N, 1, 8, 1000)`

文件建议：`preprocess_lab/src/steps/resample_zscore.py`

## 参考示例

```python
import numpy as np


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
    arr = np.stack(trials, axis=0)      # (N, 1000, 8)
    arr = np.transpose(arr, (0, 2, 1))  # (N, 8, 1000)
    return arr[:, None, :, :].astype(np.float32)
```

## 自测建议

```python
x = np.random.randn(1000, 8).astype(np.float64) * 50 + 10
z = trial_zscore(x)
assert z.shape == (1000, 8)
assert np.allclose(z.mean(axis=0), 0, atol=1e-6)
assert np.allclose(z.std(axis=0), 1, atol=1e-5)

X = to_eegnet_tensor([z, z])
assert X.shape == (2, 1, 8, 1000)
```

## 与旧工程的区别

| | 旧 `MI_model` 常见做法 | 本流水线 |
|--|------------------------|----------|
| 标准化 | 训练集上算通道 mean/std | **每个 trial 自己** mean/std |
| 张量 | 可能是 `(N,22,T)` 等 | **`(N,1,8,1000)`** |

## 常见坑

- 在整段连续信号上做 z-score，而不是单试次窗。
- `std` 用了 `ddof=1` 与文档不一致时，只要全流水线统一并在记录里写清即可；默认 `np.std`（ddof=0）即可。
- `transpose` 轴弄反变成 `(N,1,1000,8)`。
- 漏了中间的空通道维 `1`。

## 验收清单

- [ ] z-score 后每通道 mean≈0、std≈1
- [ ] `to_eegnet_tensor` 输出 `(N,1,8,1000)`、`float32`
- [ ] 无 NaN（注意全零通道用 `eps` 保护）

## 提交检查时附上

两函数代码 + 真实若干 trial 的 `mean/std` 检查 + `X.shape`。
