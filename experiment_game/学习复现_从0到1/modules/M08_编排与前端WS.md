# M08 · Session 编排 + 薄 WS（详细：参考谁、怎么写）

---

## 一句话目标

把 M01–M07 **串成一场会话**（CLI 必做）。WS/网页可选。

---

## 原版精读表

| 步 | 文件 | 搜 / 读 | 看懂即停 |
|----|------|---------|----------|
| 1 | `experiment_game/experiment/session.py` | `create_session_dir` | 目录怎么建 |
| 2 | `experiment_game/experiment/session_v3.py` | 文件头 docstring | 会话职责 |
| 3 | 同上 | 搜 `for` + `trial` / `run_round` / `TrialStateMachine` | **主循环在哪几行** |
| 4 | 同上 | 搜 `judge` / `tracker` / `MiTrialTracker` | 判定与计分何时调用 |
| 5 | `tools/open_operator.py` | 启动 WS/HTTP | 迷你版可更简 |
| 6 | `experiment_game/experiment/ws_bridge.py` | 类 `WsBridge`：`broadcast` / `start` | 广播 API |
| 7 | `docs/ws_protocol.md` | 前两节 + 搜 `session_start` 一类 | 消息 type 举例 |
| 8 | `web/js/operator.js` | 搜 `start` | 前端发什么 JSON |
| 9 | `orchestrator.py` | **只搜** 与 start 相同的 type 字符串 → `_handle_...` | 路由器长什么样 |

**禁止**：从头到尾读 orchestrator。

---

## 你要写的签名草稿

### `session_dir.py`

```python
def create_session_dir(root, subject, session_name) -> Path:
    # 建文件夹；返回路径
    ...
```

对照：`create_session_dir`。

### `session_v3.py`（核心，先伪代码填实）

```python
def run_v3_session(
    *,
    out_dir: Path,
    timing,
    n_blocks: int,
    trials_per_block: int,
    labels: list[str],   # 或内部生成
    source,
    model,               # Fake 即可
):
    logger = EventLogger(out_dir / "events.jsonl")
    # 写 run_config.json
    # start capture
    # for each trial:
    #     run_trial(... emit=logger.append)
    #     for each window end time:
    #         out = inference.judge(...)
    #         tracker.add_window(out["p_three"])
    #     result = tracker.finalize()
    #     记录得分
    # stop; alignment.verify; 写报告
```

对照：`session_v3` 主循环——只学**顺序**，用你自己的函数名。

### `tools/run_synthetic_session.py`

```python
# argparse: --subject --out
# 调用 run_v3_session
# 第一版可用超短：1 block × 2 trials，timing 改小（仅调试）或逻辑钟
```

### （可选）`ws_bridge.py` + `serve_mini_operator.py`

```python
# 收到 {"type":"start_session", ...} → 线程里 run_v3_session
# broadcast {"type":"trial_score", "score":...}
```

对照：`WsBridge.broadcast`；消息 type 自定，README 列出来。

### 测试

`test_session_smoke.py`：逻辑钟 + FakeModel + 2 trials，断言 `events.jsonl` 存在且含 `mi_start`。

---

## 逐步仿写（强烈建议这个顺序）

1. **无 EEG**：只 `run_trial` × N，写 events（验证编排）。  
2. 加 SyntheticSource + CSV（验证采集落盘）。  
3. 加 Fake judge + F5（验证得分）。  
4. 加 alignment。  
5. 最后才加 WS。

每一步都留在 git 或笔记里「当前能跑什么」。

---

## 常见坑

| 坑 | 做法 |
|----|------|
| 一开始就抄 operator 全 UI | 先 CLI |
| 真实 sleep 跑 2×18 测单测 | 缩短或逻辑钟 |
| 业务写在 JS | 分数只在 Python 算 |

---

## 验收

- [ ] CLI 跑通并落盘  
- [ ] 能讲清点击/CLI → session → scoring  
- [ ] 烟雾测试绿 |

## 下一模块

[`M09_落盘与alignment.md`](M09_落盘与alignment.md)
