# M04 · 试次状态机（详细：参考谁、怎么写）

---

## 一句话目标

自己写 `run_trial`：按配置走完 Rest→prep→cue→MI→ITI，并 `emit` 事件。

---

## 现行数字（先打开核对）

打开 **`experiment_game/config/protocol.yaml`** 里 `timing_v3`（以文件为准）：

| 键 | 典型值 | 含义 |
|----|--------|------|
| `inter_trial_rest_s` | 4.0 | Rest |
| `prep_s` | 2.0 | 准备 |
| `cue_s` | 1.0 | Cue 展示 |
| `imagine_s` | 4.0 | MI |
| `iti_s` | 3.0 | ITI |

你的 `mi_scratch/config/protocol.yaml` 应已复制同值。

---

## 原版精读表

| 步 | 文件 | 搜 / 读 | 看懂即停 |
|----|------|---------|----------|
| 1 | `config/protocol.yaml` + `config/v3_session.yaml` | `timing` / `prep_s`… | 数字从哪来 |
| 2 | `experiment_game/experiment/v3_config.py` | 加载 yaml 的函数/类 | 配置如何进对象 |
| 3 | `experiment_game/experiment/openbmi_align_config.py` | 文件头 + 与 timing 相关常量 | Align 共享配置 |
| 4 | `experiment_game/experiment/trial_v2.py` | 类 `TrialTimingV2` | 字段有哪些 |
| 5 | 同上 | 类 `TrialStateMachineV2` | **不要通读**；搜 `rest` / `prep` / `cue` / `imagine` / `iti` |
| 6 | 同上 | 搜 `rest_start` 或 `mi_start` 或 `emit`/`marker` | 事件在哪打 |
| 7 | `experiment_game/experiment/session_v3.py` | 搜 `TrialStateMachine` 或 `run_round` | 会话如何调用（只看调用行） |

小白策略：原版状态机很长。你的第一版用「阶段列表 + for 循环」即可，不必抄类结构。

---

## 你要写的签名草稿

### `timing.py`

```python
from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class TrialTiming:
    rest_s: float
    prep_s: float
    cue_s: float
    imagine_s: float
    iti_s: float

def load_timing(path: Path) -> TrialTiming:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    t = cfg["timing_v3"]
    return TrialTiming(
        rest_s=float(t["inter_trial_rest_s"]),
        prep_s=float(t["prep_s"]),
        cue_s=float(t["cue_s"]),
        imagine_s=float(t["imagine_s"]),
        iti_s=float(t["iti_s"]),
    )
```

### `trial_fsm.py`（逻辑钟版，好测）

```python
def run_trial(label: str, timing: TrialTiming, emit, *, t0: float = 0.0) -> float:
    """
    emit(name, t, **fields)
    返回试次结束时间 t
    不 sleep：用 t 累加时长（单测友好）
    """
    t = t0
    emit("rest_start", t, label=label)
    t += timing.rest_s
    emit("rest_end", t, label=label)
    emit("prep_start", t, label=label)
    t += timing.prep_s
    emit("cue", t, label=label)
    # 按你对照原版的锚点策略二选一，写进注释：
    # A) mi_start 与 cue 同刻；B) cue 后再 +cue_s 才 mi_start
    # ★ 必须与 M05 切窗锚点一致，并对照现行 session/trial 代码
    emit("mi_start", t, label=label)  # 示例：先按「与 cue 同刻」；若原版不同请改
    t += timing.imagine_s
    emit("mi_end", t, label=label)
    emit("iti_start", t, label=label)
    t += timing.iti_s
    return t
```

**重要**：上面 `mi_start` 与 `cue` 的关系，请你打开原版 `trial_v2` 里实际打点顺序后改到与原版一致。不会判断时：对比对照场 `events.jsonl` 里同一 trial 的 `cue` 与 `mi_start` 时间是否相同。

### 测试

```python
def test_order():
    names = []
    def emit(n, t, **kw):
        names.append(n)
    run_trial("Left", timing, emit)
    assert names[:3] == ["rest_start", "rest_end", "prep_start"]
```

再用假 clock 测：`rest_end.t - rest_start.t == rest_s`。

---

## 逐步仿写

1. 写 `load_timing`，打印五个数字，与 yaml 一致。  
2. 打开对照 `events.jsonl`，抄一条完整 trial 的事件名顺序。  
3. 把顺序写成测试。  
4. 实现 `run_trial` 到绿。  
5. 打开 `TrialStateMachineV2` 里搜到的 marker 行，确认你没漏事件。

---

## 常见坑

| 坑 | 做法 |
|----|------|
| 单测用 `time.sleep(4)` | 用时间累加 |
| cue / mi_start 与原版不一致 | 以对照 jsonl 为准改 |
| prep 算进 Rest 分 | 概念上分开；计分在 M07 |

---

## 验收

- [ ] 能画出时间轴（含现行 cue_s）  
- [ ] 顺序/时长测试绿  
- [ ] README 写明切窗锚点用哪个事件 |

## 下一模块

[`M05_在线切窗与预处理.md`](M05_在线切窗与预处理.md)
