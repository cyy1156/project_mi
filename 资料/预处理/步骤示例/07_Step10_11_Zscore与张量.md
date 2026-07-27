# Step 10–11：Trial Z-score + 张量规整

## 目标

1. **Trial-wise Z-score**：每个试次、每个通道，在分类窗（2a：2 s / 500 点）上独立标准化。
2. 堆成 EEGNet 常用形状：`(N, 1, 8, T)`（2a：`T=500`）

文件：`preprocess_lab/src/common/steps/resample_zscore.py`

## 参考示例

```python
import numpy as np


def trial_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: (T, 8) → 同形状，每通道独立标准化。"""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def to_eegnet_tensor(trials: list[np.ndarray]) -> np.ndarray:
    """
    trials: 每个元素 (T, 8)
    输出: (N, 1, 8, T)
    """
    arr = np.stack(trials, axis=0)      # (N, T, 8)
    arr = np.transpose(arr, (0, 2, 1))  # (N, 8, T)
    return arr[:, None, :, :].astype(np.float32)
```

### 自测（2a）

```python
x = np.random.randn(500, 8).astype(np.float64) * 50 + 10
z = trial_zscore(x)
assert z.shape == (500, 8)
assert np.allclose(z.mean(axis=0), 0, atol=1e-6)

X = to_eegnet_tensor([z, z])
assert X.shape == (2, 1, 8, 500)
```

## 与训练侧约定

| 项 | 说明 |
|----|------|
| 张量 | 预处理存 `(N,1,8,T)`；Dataset 再 squeeze 成 `(B,8,T)` |
| `n_times` | **固定 500**（2s@250Hz）；也可用 `X.shape[-1]` |

## 常见坑

- `transpose` 轴弄反变成 `(N,1,T,8)`。
- 仍断言 `(1,8,1000)` 却喂入现行 500 点数据。

## 验收清单

- [ ] `to_eegnet_tensor` 输出 `(N,1,8,T)`、`float32`（2a：T=500）
