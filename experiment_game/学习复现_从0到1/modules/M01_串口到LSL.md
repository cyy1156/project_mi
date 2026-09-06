# M01 · 串口到样本源 —— 怎么写 + 完整答案示例

> **作业落盘**：代码写到 `self_learing/src/self_learing/`（测试 `self_learing/tests/`）。
> **完整示例 + 路径树**：[`../examples/M01_代码示例.md`](../examples/M01_代码示例.md) —— **自己重写，勿整文件复制交差**。
> **禁止**：把示例写进现网 `experiment_game/`。


> 作业目录：`D:\MI\self_learing\`  
> 目标：写出「8 导 × 250Hz 的合成样本水龙头」。  
> 下面给出**可直接对照的完整答案**（包名已是 `self_learing`）。请先读懂再敲进你的文件；不要不看就整份粘贴交差。

---

## 0. 先搞懂：本模块在整条链的哪里

```text
【本模块】SyntheticSource 造假 EEG
    ↓
【M02】RingBuffer + Bus + CSV
    ↓
后面才是试次 / 模型 / 计分
```

原版里真机是：`COM → AcquisitionFacade → LSL`。  
你第一版**不接串口**，只做一个接口类似的合成泵，后面模块才能接着写。

**原版请打开看一眼（懂 start/stop 即可，不要抄驱动）：**

| 文件 | 看什么 |
|------|--------|
| `experiment_game/core/channel_layout.py` | `DEVICE_CHANNEL_LABELS` 那 8 个名字 |
| `experiment_game/acquisition/service.py` | `AcquisitionFacade` 的 `start` / `stop`；`filter_enabled` |

---

## 1. 你要创建 / 改哪些文件

在 `D:\MI\self_learing\` 下最终应有：

```text
self_learing/
  README.md                          ← 你写 3 句说明
  src/
    self_learing/
      __init__.py                    ← 已有可空
      channels.py                    ← 【答案见 §2】你可能还没有
      source_base.py                 ← 【可选，答案见 §3】
      source_synthetic.py            ← 【答案见 §4】你可能已有草稿
  tests/
    test_smoke.py                    ← 已有
    test_synthetic_source.py         ← 【答案见 §5】你要新建
```

---

## 2. 完整答案：`src/self_learing/channels.py`

**作用**：通道名、采样率、导联数。顺序必须和原版一模一样。

```python
"""通道常量 —— 与 experiment_game/core/channel_layout.py 的 DEVICE_CHANNEL_LABELS 一致。"""

from __future__ import annotations

from typing import List

CHANNEL_NAMES: List[str] = [
    "FC3",
    "C3",
    "CP3",
    "CZ",
    "CPZ",
    "FC4",
    "C4",
    "CP4",
]

FS_HZ: float = 250.0
N_CH: int = len(CHANNEL_NAMES)  # 8
```

**怎么自己写**：打开原版 `channel_layout.py`，把那 8 个字符串按顺序敲进列表。

---

## 3. 完整答案（可选）：`src/self_learing/source_base.py`

**两种写法任选其一**（详见 [`examples/M01_代码示例.md`](../examples/M01_代码示例.md) §2）：

- **继承版（推荐）**：`BaseSampleSource(ABC)` + 子类 `SyntheticSource` 实现 `iter_n` / `_pull_one_sample`  
- **Protocol 版**：只规定接口，不强制继承  

下面保留 Protocol 最小版；若你选继承，请按示例文档 **§2B–§2C** 自己敲父类与子类。

```python
"""样本源接口（可选）。"""

from __future__ import annotations

from typing import Iterator, Protocol, Tuple

import numpy as np


class SampleSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def samples(self) -> Iterator[Tuple[float, np.ndarray]]:
        """yield (t_seconds, x)，要求 x.shape == (8,)"""
        ...
```

**继承版入口**：打开 `examples/M01_代码示例.md` → 「文件 2 · 继承版」→ 先写 `source_base.py`，再让 `source_synthetic.py` 里 `class SyntheticSource(BaseSampleSource)`。

---

## 4. 完整答案：`src/self_learing/source_synthetic.py`

**作用**：造假脑电。测试用 `iter_n`（不 sleep）；`start`/`stop` 对应原版开关采集。

```python
"""合成样本泵 —— M01。

对照原版 AcquisitionFacade：
  - start/stop：开关「是否出数」
  - 本文件不碰 COM/LSL，只制造假数据给后面 Buffer 用
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from self_learing.channels import FS_HZ, N_CH


class SyntheticSource:
    def __init__(
        self,
        fs: float = FS_HZ,
        n_ch: int = N_CH,
        seed: Optional[int] = 0,
    ) -> None:
        if int(n_ch) != N_CH:
            raise ValueError(f"n_ch 必须为 {N_CH}，收到 {n_ch}")
        self.fs = float(fs)
        self.n_ch = int(n_ch)
        self._rng = np.random.default_rng(seed)  # 固定种子，测试可重复
        self._running = False

    def start(self) -> None:
        """对应原版 Facade.start。"""
        self._running = True

    def stop(self) -> None:
        """对应原版 Facade.stop。"""
        self._running = False

    def iter_n(
        self, n: int, start_t: float = 0.0
    ) -> Iterator[Tuple[float, np.ndarray]]:
        """一次产出 n 个样本，不 sleep。

        每个元素：
          t: float，秒
          x: np.ndarray，shape == (8,)，dtype float64
        """
        n = int(n)
        for i in range(n):
            t = float(start_t) + i / self.fs
            x = self._rng.standard_normal(self.n_ch).astype(np.float64)
            yield t, x

    def samples(self) -> Iterator[Tuple[float, np.ndarray]]:
        """无限流（实时风格）。单测请优先用 iter_n。"""
        if not self._running:
            self.start()
        i = 0
        while self._running:
            t = i / self.fs
            x = self._rng.standard_normal(self.n_ch).astype(np.float64)
            yield t, x
            i += 1
```

### 关键几行在干什么

| 代码 | 含义 |
|------|------|
| `t = start_t + i / self.fs` | 第 i 个点的时间；250Hz 时相邻点差 0.004 秒 |
| `standard_normal(8)` | 造 8 个数当假脑电 |
| `yield t, x` | 一次交出一个样本，外面用 `for` 接收 |
| `_running` | `start`/`stop` 开关，给以后实时循环用 |

---

## 5. 完整答案：`tests/test_synthetic_source.py`

**新建这个文件**（和 `test_smoke.py` 放一起）。

```python
import sys
from pathlib import Path

# 让 Python 能 import 到 src/self_learing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from self_learing.channels import CHANNEL_NAMES, N_CH
from self_learing.source_synthetic import SyntheticSource


def test_iter_n_count_and_shape():
    src = SyntheticSource(fs=250.0, n_ch=8, seed=0)
    xs = list(src.iter_n(250))
    assert len(xs) == 250
    t0, x0 = xs[0]
    assert isinstance(t0, float)
    assert isinstance(x0, np.ndarray)
    assert x0.shape == (8,)
    assert x0.dtype == np.float64


def test_time_spacing():
    src = SyntheticSource(fs=250.0, seed=1)
    xs = list(src.iter_n(3, start_t=1.0))
    assert abs(xs[0][0] - 1.0) < 1e-9
    assert abs(xs[1][0] - (1.0 + 1.0 / 250.0)) < 1e-9


def test_channel_names_match_openbmi_order():
    assert CHANNEL_NAMES == [
        "FC3",
        "C3",
        "CP3",
        "CZ",
        "CPZ",
        "FC4",
        "C4",
        "CP4",
    ]
    assert N_CH == 8


def test_start_stop_flags():
    src = SyntheticSource()
    src.start()
    assert src._running is True
    src.stop()
    assert src._running is False
```

---

## 6. 按这个顺序操作（照做）

1. **新建** `src/self_learing/channels.py`，粘贴 §2 内容并保存。  
2. **核对 / 覆盖** `source_synthetic.py`：应与 §4 一致（你若已有草稿，对照改到一样即可）。  
3. **新建** `tests/test_synthetic_source.py`，粘贴 §5。  
4. 打开终端运行：

```powershell
cd D:\MI\self_learing
python -m pytest tests/test_synthetic_source.py -q
```

5. 若报错 `No module named self_learing.channels` → 说明 `channels.py` 路径不对或没保存。  
6. 全绿后，在根目录写 `README.md` 三句（见 §7）。

---

## 7. README 三句（答案文案示例）

在 `D:\MI\self_learing\README.md` 里写（可照抄后再用自己的话改一版）：

```text
## M01 对照说明

1. 原版 AcquisitionFacade.start/stop：打开/关闭板卡到 LSL 的采集。
2. 我的 SyntheticSource：不接板，只提供同样的「持续 8 导样本」接口，供后面 Buffer 使用。
3. 采集侧 filter_enabled 默认常关，避免和模型侧预处理双重滤波。
```

---

## 8. 验收清单（全勾才算 M01 完成）

- [ ] 存在 `channels.py`，8 名顺序正确  
- [ ] `pytest tests/test_synthetic_source.py -q` **全绿**  
- [ ] 能说出：串口逻辑在原版 `acquisition`；你的 Source 是实验侧样本入口  
- [ ] README 有三句对照  
- [ ] **还没写** RingBuffer / Bus（那是 M02）

完成后回复「M01 做完了」，再进入 M02。

---

## 9. 常见报错

| 报错 | 原因 | 处理 |
|------|------|------|
| `No module named 'self_learing.channels'` | 缺少 `channels.py` | 按 §2 新建 |
| `No module named 'self_learing'` | `sys.path` 没加 `src` | 测试文件顶部按 §5 加两行 |
| `shape == (8,)` 失败 | `x` 写成了 list 或 `(8,1)` | 用 `astype` 后的一维 `ndarray` |
| 测试很慢 | 用了 `sleep` | 只用 `iter_n`，不要 sleep |

---

## 下一模块

[`M02_RingBuffer与EEGBus.md`](M02_RingBuffer与EEGBus.md)  
（同样会有完整答案示例；先把本模块测绿。）
