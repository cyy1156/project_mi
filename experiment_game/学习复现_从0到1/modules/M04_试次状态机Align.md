# M04 · 试次状态机（OpenBMI-Align）

## 目标

把「一个 trial 的一生」对应到配置字段与状态机代码。

## 时序（背下来）

```text
[Rest 4s] → [prep 2s] → [Cue = MI onset] → [MI 4s] → [ITI 3s]
```

| 配置键 | 值 | 文件 |
|--------|-----|------|
| `inter_trial_rest_s` | 4.0 | `config/v3_session.yaml` · `protocol.yaml` |
| `prep_s` | 2.0 | 同上 |
| `cue_s` | 0.0 | Cue 无额外展示秒 |
| `imagine_s` | 4.0 | MI 固定时长 |
| `iti_s` | 3.0 | 试次末缓冲 |

v3 默认：`blocks: 2` · `trials_per_block: 18`。

## 对应代码

| 内容 | 路径 | 看什么 |
|------|------|--------|
| 状态机 | `experiment/trial_v2.py` | `TrialTimingV2` · `TrialStateMachineV2` · `run_round` |
| 共享 Align 常量 | `experiment/openbmi_align_config.py` | 判定时刻网格等 |
| 会话循环 | `experiment/session_v3.py` | 如何对每个 trial 调状态机 + 判定 |
| 配置加载 | `experiment/v3_config.py` | YAML → 对象 |
| 冻结口述 | `docs/框架冻结确认_20260829.md` 现行时序节 | F1/F2 |

## 精读顺序

1. `v3_session.yaml` 时序字段（5 分钟）  
2. `trial_v2.py`：找 `rest` / `prep` / `cue` / `imagine` / `iti` 等待或 sleep 逻辑  
3. `session_v3.py`：搜索 `TrialStateMachine` 或 `run_round` 调用处（不要通读全文件）

## 动手题

1. 默写时序，对照 yaml 无误。  
2. 在状态机里找到打 `rest_start` / `mi_start` 的位置（或调用 Marker 处）。  
3. 解释：为什么说「prep 不算 Rest 计分」。

## 验收

- [ ] 不看文档能画出 trial 时间轴  
- [ ] 能指出 `cue_s=0` 与「Cue=mi_start」的关系  

## 下一模块

[`M05_在线切窗与预处理.md`](M05_在线切窗与预处理.md)
