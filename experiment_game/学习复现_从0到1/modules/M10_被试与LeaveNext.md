# M10 · Leave-Next 调度（详细：参考谁、怎么写）

---

## 一句话目标

自己写：**选哪些场训练、hold 哪一场、循环跑 FT、写 summary**。  
真正的梯度更新可调用原版 `pipeline/finetune.py`。

---

## 原版精读表

| 步 | 文件 | 搜 / 读 | 看懂即停 |
|----|------|---------|----------|
| 1 | `docs/被试登录与按session微调方案_20260827.md` | 前半目录约定 | subjects 树 |
| 2 | 冻结表 | F7 在线 FT 关；F8 promote；F9 replay；F11 只要 v3 | 政策 |
| 3 | `experiment_game/tools/run_leave_next_e1f_task_ramp.py` | **文件头注释** + `RAMP_` 列表 | 爬坡长什么样 |
| 4 | 同上 | `_list_v3_sessions` / 过滤函数 | 如何筛 v3 |
| 5 | 同上 | `_ramp_for_subject` | 缺场如何 skip |
| 6 | `experiment_game/pipeline/finetune.py` | 搜 `def run_subject_finetune` 或主入口 | **只读签名+docstring** |
| 7 | `tools/ft_subject_from_v3.py` | argparse | CLI 如何传参 |
| 8 | 一场现成 `*leave_next*summary.json` | 字段 | 你的 summary 子集对齐 |
| 9 | `orchestrator.py` | 只搜 `_handle_finetune` | 操作台如何触发（了解即可） |

---

## 你要写的签名草稿

### `ft_filters.py`

```python
def is_trainable_session(meta: dict) -> bool:
    if meta.get("phase_mode") != "v3_session":
        return False
    if meta.get("complete") is False:
        return False
    return True
```

### `leave_next.py`

```python
def build_ramp(session_keys: list[str]) -> list[tuple[list[str], str, bool]]:
    """
    例如 keys=[w01..w06]
    返回 [([w01],w02,True), ([w01,w02],w03,True), ...]
    对照原版 RAMP_W；replay 布尔可先简化
    """
    out = []
    for i in range(1, len(session_keys)):
        train = session_keys[:i]
        hold = session_keys[i]
        use_replay = i < 3  # 示例：前几档 True，对照原版改
        out.append((train, hold, use_replay))
    return out
```

打开原版 `RAMP_W` / `RAMP_YCX`，把 True/False 抄准。

### `tools/run_leave_next.py`

```python
# 1. 列出 subject 下 session
# 2. 过滤
# 3. for train, hold, replay in build_ramp(...):
#        调用 FakeFT 或真 finetune
#        写 summary 一行/一个 json
```

### FakeFT（先通调度）

```python
def fake_finetune(train_dirs, hold_dir, out_dir) -> dict:
    return {"release_pass": False, "heldout_acc": 0.0, "note": "fake"}
```

真 FT：读 `run_subject_finetune` 签名，用子进程或 import 调用；失败时看原版 CLI 参数。

### 测试

- 6 个 key → 5 档；hold 分别是第 2…6 场。  
- meta incomplete 的场不会出现在 keys。  
- v4 meta 被拒。

---

## 逐步仿写

1. 只测 `build_ramp` / `is_trainable_session`（不需 GPU）。  
2. CLI + FakeFT 写出 summary 文件。  
3. 有环境再换真 finetune。  
4. promote：显式函数 `promote(run_dir, current_dir)`，gate FAIL 默认不覆盖。

---

## 常见坑

| 坑 | 做法 |
|----|------|
| 把帽检 v4 训进去 | 过滤 phase_mode |
| 半场当满场 | complete 标志 |
| 一上来就调全员 all4 GPU | 先 FakeFT |

---

## 验收

- [ ] 能画 Leave-Next 图  
- [ ] 过滤+ramp 测试绿  
- [ ] README 声明 finetune 是否复用原版 |

## 下一模块

[`M11_端到端验收.md`](M11_端到端验收.md)
