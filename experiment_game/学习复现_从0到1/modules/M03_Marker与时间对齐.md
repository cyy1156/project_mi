# M03 · EventLog + Marker + alignment（详细）

> **作业落盘**：代码写到 `self_learing/src/self_learing/`（测试 `self_learing/tests/`）。
> **完整示例 + 路径树**：[`../examples/M03_代码示例.md`](../examples/M03_代码示例.md) —— **自己重写，勿整文件复制交差**。
> **禁止**：把示例写进现网 `experiment_game/`。


---

## 一句话目标

会写 **`events.jsonl`（一行一个事件）**，并做最小时间校验。

---

## 原版精读表

| 步 | 文件 | 搜 / 读 | 停下来的标准 |
|----|------|---------|--------------|
| 1 | 你的对照场 `events.jsonl` | 打开看 5～10 行 | 知道每行有哪些键（时间、事件名、label…） |
| 2 | `experiment_game/experiment/markers.py` | `format_payload`、类 `MarkerPublisher` | 事件如何被格式化/发送 |
| 3 | 在 `experiment_game/experiment/` 下 Cursor 全局搜 `EventLogger` 或 `events.jsonl` | 跳到写入处 | 谁在 `append`/write |
| 4 | `experiment_game/experiment/alignment.py` | 搜 `verify` / `passed` / `checks` | 校验报告长什么样 |
| 5 | `docs/范式对齐_OpenBMI与fnz_v3_20260827.md` | §2 事件序 | 事件名字列表 |
| 6 | `docs/marker_spec.md` | 仅作历史参考 | **现行以 Align 事件名为准** |

---

## 你要写的签名草稿

### `events.py`

```python
import json
from pathlib import Path

class EventLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def append(self, name: str, t: float, **fields):
        row = {"name": name, "t": float(t), **fields}
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()
```

对照：原版写入字段名可能叫 `event` / `type`——**以你对照场 jsonl 实际字段为准**，保持和原版一致更易 diff。若原版用 `event`，你就用 `event`，不要自创两套。

### `markers.py`

```python
class MarkerSink:
    def __init__(self, logger: EventLogger):
        self.logger = logger
    def emit(self, name: str, t: float, **fields):
        self.logger.append(name, t, **fields)
        # 以后可加真 LSL；现在不必
```

对照：`MarkerPublisher`——你先只落盘。

### `alignment.py`

```python
def verify_events(events: list[dict], t0: float, t1: float) -> dict:
    checks = []
    ok = True
    for e in events:
        t = float(e["t"])
        passed = (t0 <= t <= t1)
        checks.append({"name": e.get("name"), "t": t, "in_range": passed})
        ok = ok and passed
    return {"passed": ok, "checks": checks}
```

对照：`alignment.py` 的 `passed` / `checks` 形状；你可以更少检查项。

### 测试

- 写 3 个事件 → 读回 → 名字顺序一致。  
- 一个事件 `t=999`，eeg 只到 `10` → `passed is False`。

---

## 逐步仿写

1. 把对照场里**真实字段名**抄到笔记本。  
2. 按真实字段写 `append`。  
3. 实现 verify。  
4. 对照范式文档，列出你支持的 name 字符串常量：

```python
REST_START = "rest_start"
# ...
```

---

## 常见坑

| 坑 | 做法 |
|----|------|
| 字段名和原版不一致 | 以对照 jsonl 为准 |
| 文件不 flush | 每行 `flush()` 或退出时 close |
| 用两套时钟乱比 | 单测里事件时间和假 eeg 范围用同一套数 |

---

## 验收

- [ ] 能说出 ≥5 个 Align 事件名  
- [ ] roundtrip + alignment 测试绿  

## 下一模块

[`M04_试次状态机Align.md`](M04_试次状态机Align.md)
