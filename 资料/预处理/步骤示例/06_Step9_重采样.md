# Step 9：重采样到 250 Hz / 窗长点数

## 目标

分类窗应对应配置的 `win_sec` 秒，统一到：

- `fs_out = 250`
- `n_out = int(win_sec * 250)`
- 形状 `(n_out, 8)`

| 数据集约定 | win_sec | n_out |
|------------|---------|-------|
| **BCI2a / Stieger / 自采 Phase4 现行** | **2.0** | **500** |

函数名历史原因仍叫 `resample_to_1000`，实际由 `win_sec` 决定点数；**默认 `win_sec=2.0` → 500 点**。

文件：`preprocess_lab/src/common/steps/resample_zscore.py`

## 参考示例

```python
from scipy.signal import resample
import numpy as np


def resample_to_1000(
    x_win: np.ndarray,
    fs_in: float,
    fs_out: float = 250.0,
    win_sec: float = 2.0,
) -> np.ndarray:
    """
    x_win: (n_times_in, 8)，应对应 win_sec 秒。
    输出: (int(win_sec*fs_out), 8)；现行默认 2s→500。
    """
    n_out = int(round(win_sec * fs_out))
    if abs(fs_in - fs_out) < 1e-6 and x_win.shape[0] == n_out:
        return x_win.astype(np.float32)
    y = resample(x_win, n_out, axis=0)
    return np.asarray(y, dtype=np.float32)
```

### 自测

```python
# 已是 250Hz / 500（2s）
a = np.random.randn(500, 8)
b = resample_to_1000(a, 250.0, win_sec=2.0)
assert b.shape == (500, 8)

# 模拟 1000Hz 的 2s → 2000 点
c = np.random.randn(2000, 8)
d = resample_to_1000(c, 1000.0, win_sec=2.0)
assert d.shape == (500, 8)
```

BCI IV 2a 本身常为 250 Hz：若分类窗已是 500 点，应走「直接返回」分支，避免无必要 resample 引入数值差。

## 常见坑

- 输入不是完整 `win_sec`（例如仍含基线多出 0.5 s）却硬 resample → 时间被压扁。
- 流水线忘传 `win_sec=2.0` 或仍按 4s 切窗，会得到错误长度。

## 验收清单

- [ ] 合法 2 s 窗 → `(500, 8)`
- [ ] 250 Hz 且已是目标点数时不改变长度
