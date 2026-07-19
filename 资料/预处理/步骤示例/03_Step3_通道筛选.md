# Step 3：通道统一筛选

## 目标

从任意通道集合中，按**固定顺序**取出 8 个目标通道。  
缺失通道：**不补零、不插值**，直接报错。

文件建议：`preprocess_lab/src/steps/select_channels.py`

## 固定顺序（禁止按字母或原文件顺序重排）

```text
索引:  0    1    2    3     4     5     6     7
名称: Cz   C3   C4   CP3   FC4   FC3   CP4   CPz
```

## 参考示例

```python
import numpy as np

TARGET_CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]


def select_channels(x: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """
    x: (n_times, n_ch)
    返回: (n_times, 8)，列顺序 = TARGET_CHANS
    """
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    missing = [c for c in TARGET_CHANS if c not in name_to_idx]
    if missing:
        raise KeyError(f"缺少目标通道: {missing}；可用: {ch_names}")
    idx = [name_to_idx[c] for c in TARGET_CHANS]
    return x[:, idx]
```

## 自测建议

```python
x8 = select_channels(eeg.x, eeg.ch_names)
assert x8.shape == (eeg.x.shape[0], 8)
# 验证顺序：抽一列对比原通道
for j, name in enumerate(TARGET_CHANS):
    old_j = eeg.ch_names.index(name)
    assert np.allclose(x8[:, j], eeg.x[:, old_j])
print("OK order", TARGET_CHANS)
```

故意缺通道：

```python
try:
    select_channels(eeg.x[:, :5], eeg.ch_names[:5])
except KeyError as e:
    print("expected:", e)
```

## 特殊说明：BCI IV 2b

只有 C3/Cz/C4，**无法**满足 8 通道标准。正确做法是：排除该数据集，或另开「低导联」实验分支——**不要偷偷补零**。

## 常见坑

- 按字母排序通道名 → 顺序错。
- 通道名大小写不一致（`CZ` vs `Cz`）导致误报缺失。
- 返回 `(8, T)` 而不是 `(T, 8)`。

## 验收清单

- [ ] 输出恒为 `(T, 8)`
- [ ] 列顺序严格等于 `TARGET_CHANS`
- [ ] 缺通道时抛出清晰错误
- [ ] 与原 22 通道对应列数值一致（allclose）

## 提交检查时附上

函数代码 + A01 上 `x8.shape` + 顺序断言通过的打印。
