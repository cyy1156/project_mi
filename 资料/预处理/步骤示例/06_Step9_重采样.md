# Step 9：重采样到 250 Hz / 1000 点

## 目标

分类窗应对应**正好 4 秒**，统一到：

- `fs_out = 250`
- `n_out = 1000`
- 形状 `(1000, 8)`

文件建议：放在 `preprocess_lab/src/steps/resample_zscore.py`（可与 Step10 同文件）

## 参考示例

```python
import numpy as np
from scipy.signal import resample


def resample_to_1000(
    x_win: np.ndarray,
    fs_in: float,
    fs_out: float = 250.0,
) -> np.ndarray:
    """
    x_win: (n_times_in, 8)，应对应正好 4 秒。
    输出: (1000, 8)
    """
    n_out = int(4.0 * fs_out)  # 1000
    if abs(fs_in - fs_out) < 1e-6 and x_win.shape[0] == n_out:
        return x_win.astype(np.float32)
    y = resample(x_win, n_out, axis=0)
    return np.asarray(y, dtype=np.float32)
```

## 自测建议

```python
# 已是 250Hz / 1000
a = np.random.randn(1000, 8)
b = resample_to_1000(a, 250.0)
assert b.shape == (1000, 8)

# 模拟 1000Hz 的 4s → 4000 点
c = np.random.randn(4000, 8)
d = resample_to_1000(c, 1000.0)
assert d.shape == (1000, 8)
assert d.dtype == np.float32
```

BCI IV 2a 本身常为 250 Hz：若分类窗已是 1000 点，应走「直接返回」分支，避免无必要 resample 引入数值差。

## 常见坑

- 输入不是完整 4 s（例如仍含基线 4.5 s）却硬 resample 到 1000 → 时间被压扁。
- `axis` 写错（对通道轴 resample）。
- 输出忘了固定 `float32`（与后续张量约定不一致时再统一也可，但建议尽早一致）。

## 验收清单

- [ ] 任意合法 4 s 窗 → `(1000, 8)`
- [ ] 250 Hz 且已是 1000 点时不改变长度
- [ ] 无 NaN

## 提交检查时附上

函数代码 + 对真实一试次 `win.shape` → `out.shape` 的打印。
