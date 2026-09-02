# M02 · RingBuffer + Bus + CSV（详细：参考谁、怎么写）

---

## 一句话目标

**一个水龙头进 Buffer，Bus 通知所有人**（CSV、以后推理、波形）。

---

## 原版精读表（严格按序，每次只读点名方法）

| 步 | 文件 | 搜这个名字 | 读完应能回答 |
|----|------|------------|--------------|
| 1 | `experiment_game/experiment/live_capture.py` | 类 `LiveEegCapture` | 谁和谁被组装在一起 |
| 2 | `experiment_game/runtime/eeg_bus.py` | 类 `EEGBus`：`subscribe`、`publish` | 扇出怎么做 |
| 3 | 同上 | `EegSubscriber` Protocol | 订阅者要有什么方法 |
| 4 | `experiment_game/experiment/inference_v2.py` | 类 `RingBuffer` | — |
| 4a | 同上 | 只读 `__init__`、`push` | 样本怎么进圈 |
| 4b | 同上 | 只读 `window_ending_at` | （了解）以后怎么按时间取窗；本周可不实现完 |
| 5 | `experiment_game/runtime/csv_recorder.py` | 类名里带 Recorder / write 的方法 | CSV 谁在写 |
| 6 | （可略）`runtime/eeg_health.py` | 文件头注释 | stall 是什么 |

**禁止本周**：读完整 `InferenceService`。

---

## 你要写的文件与签名草稿

### 1）`eeg_bus.py`（先写这个，最简单）

```python
class EegBus:
    def __init__(self):
        self._subs = []

    def subscribe(self, fn):
        """fn(t: float, x: np.ndarray) -> None"""
        self._subs.append(fn)

    def publish(self, t: float, x: np.ndarray) -> None:
        for fn in list(self._subs):
            fn(t, x)
```

对照原版：`EEGBus.publish` / `subscribe`——你的可以更简单（同步回调即可）。

### 2）`ring_buffer.py`

```python
class RingBuffer:
    def __init__(self, n_ch=8, fs=250.0, capacity_s=30.0):
        ...
    def push(self, t: float, x: np.ndarray) -> None:
        """写入一个样本；满了丢最旧"""
        ...
    def __len__(self) -> int:
        ...
    # 本周可选：
    def get_latest(self, n: int):
        """返回最近 n 个样本，shape (n, 8) 或 (8, n)，自己定一种并写进文档"""
        ...
```

对照原版：`RingBuffer.push`。原版一次可推多点；你**先支持单点 push** 就够。

### 3）`csv_recorder.py`

```python
class CsvRecorder:
    def __init__(self, path, channel_names):
        ...
    def on_sample(self, t, x):  # 订阅 Bus
        # 第一行写 header；以后每行: t, ch1, ch2, ...
        ...
    def close(self):
        ...
```

对照：`csv_recorder.py` 里「打开文件、写头、写行」的逻辑。

### 4）`live_capture.py`（组装）

```python
class LiveCapture:
    def __init__(self, source, buffer, bus, recorder):
        ...
    def run_n_samples(self, n: int) -> None:
        for t, x in source.iter_n(n):
            buffer.push(t, x)
            bus.publish(t, x)   # recorder 已 subscribe(bus)
```

对照：`LiveEegCapture` 的组装关系（不必抄线程模型）。

### 5）测试

`tests/test_bus_fanout.py`：

```python
def test_two_subscribers_same_values():
    bus = EegBus()
    a, b = [], []
    bus.subscribe(lambda t, x: a.append((t, x.copy())))
    bus.subscribe(lambda t, x: b.append((t, x.copy())))
    x = np.arange(8.0)
    bus.publish(0.0, x)
    assert len(a) == len(b) == 1
    assert np.allclose(a[0][1], b[0][1])
```

`tests/test_csv_rows.py`：跑 `run_n_samples(500)`，读 CSV，行数（不含表头）== 500。

---

## 逐步仿写

1. 只实现 Bus + 扇出测试（10～20 行）。  
2. 实现 Buffer.push（可用 `collections.deque(maxlen=...)`）。  
3. Recorder 订阅 Bus；LiveCapture 串起来。  
4. 打开原版 `push`，对比：你是否也在 push 时（或 publish 时）通知总线？原版是 `attach_bus`——你可以在 `push` 末尾 `bus.publish`，或像上面在 LiveCapture 里 publish，**二选一写清**。  
5. 用 M01 的 SyntheticSource 出一份真 CSV，打开看通道名。

---

## 常见坑

| 坑 | 做法 |
|----|------|
| 订阅者里保存了 `x` 的引用，后面被改 | `x.copy()` |
| CSV 忘写 header | 第一行写通道名 |
| 又写了一个「直接读 LSL」给推理 | 禁止；推理只能读同一 Buffer |

---

## 验收

- [ ] 能讲「为何不能双路拉流」  
- [ ] bus / csv 测试绿  
- [ ] 指出原版 `EEGBus`、`RingBuffer.push`、`LiveEegCapture` 与你的文件对应 |

## 下一模块

[`M03_Marker与时间对齐.md`](M03_Marker与时间对齐.md)
