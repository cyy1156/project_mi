# Step 4–5：CAR + Notch + Bandpass

## 目标

对连续信号（已选 8 通道）做：

1. **CAR**（共平均参考）
2. **Notch 50 Hz**
3. **Bandpass 8–30 Hz**

文档硬性要求：**CAR 在滤波之前**。

文件建议：`preprocess_lab/src/steps/filter_car.py`

## 参考示例

```python
import numpy as np
import mne


def car_reference(x: np.ndarray) -> np.ndarray:
    """x: (n_times, n_ch) → 每时刻减去全通道均值。"""
    return x - x.mean(axis=1, keepdims=True)


def notch_and_bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Notch 50 Hz + Bandpass 8–30 Hz。
    mne.filter 期望 (n_ch, n_times)，注意转置。
    """
    data = x.T  # (n_ch, n_times)
    data = mne.filter.notch_filter(data, Fs=fs, freqs=50.0, verbose=False)
    data = mne.filter.filter_data(
        data, sfreq=fs, l_freq=8.0, h_freq=30.0, verbose=False
    )
    return data.T  # 回到 (n_times, n_ch)
```

推荐封装顺序：

```python
def car_then_filter(x: np.ndarray, fs: float) -> np.ndarray:
    x = car_reference(x)
    x = notch_and_bandpass(x, fs)
    return x
```

## 自测建议

```python
x8 = select_channels(eeg.x, eeg.ch_names)
x_car = car_reference(x8)
# CAR 后每时刻通道均值应接近 0
assert np.allclose(x_car.mean(axis=1), 0, atol=1e-6)

x_f = notch_and_bandpass(x_car, eeg.fs)
assert x_f.shape == x8.shape
assert np.isfinite(x_f).all()

# 可选：画 C3（索引 1）滤波前后一段波形，肉眼看工频/漂移是否压下去
```

## 三者别混（对照表）

| 操作 | 改什么 | 用哪段数据 | 顺序 |
|------|--------|------------|------|
| CAR | 空间公共噪声 | **连续整段** | 滤波前 |
| 基线校正 | 直流偏移 | Epoch `[-0.5,0]` | 切 epoch 后（Step6–8） |
| Trial Z-score | 幅值分布 | 分类窗 4 s | 重采样后（Step10） |

## 常见坑

- 先滤波再 CAR → 不符合本标准。
- 忘了转置，把 `(T, C)` 直接丢给 mne → 滤波维度错。
- Notch 频率写成 60（数据若是中国/欧标 50 Hz 场景）。
- 在 epoch 上分别滤波却对连续段用不同参数（本流水线是**连续段滤波后再切**）。

## 验收清单

- [ ] 函数输入输出均为 `(T, 8)`
- [ ] CAR 后各时刻均值 ≈ 0
- [ ] 滤波后无 NaN/Inf
- [ ] 代码顺序明确：CAR → Notch → Bandpass

## 提交检查时附上

完整函数 + CAR 均值检查打印 +（可选）滤波前后均值/方差对比。
