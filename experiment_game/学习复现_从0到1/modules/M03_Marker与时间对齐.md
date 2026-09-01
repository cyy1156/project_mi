# M03 · Marker、events 与时间对齐

## 目标

分清三条时间相关产物：

| 产物 | 干什么 |
|------|--------|
| LSL Markers | 实时流上的事件，便于外部工具对齐 |
| `events.jsonl` | 本会话权威事件日志（落盘） |
| alignment/ | 事后校验「事件是否落在 eeg 时间范围内」等 |

## 对应代码

| 内容 | 路径 | 看什么 |
|------|------|--------|
| Marker 发布 | `experiment/markers.py` | `MarkerPublisher` |
| 事件落盘 | 搜 `EventLogger` / `events.jsonl`（多在 session / orchestrator 路径） | 写入格式 |
| 对齐校验 | `experiment/alignment.py` | `verify_report` 生成逻辑 |
| 历史 Marker 表 | `docs/marker_spec.md` | **Phase1 历史**；现行以 Align 事件名为准 |
| 范式事件序 | `docs/范式对齐_OpenBMI与fnz_v3_20260827.md` §2 | `rest_start/end` · `prep_start` · `cue`/`mi_start` · `mi_end` · `iti_start` |

## 关键点

- **墙钟**：以 LSL 时钟为准，不要混用本机 `time.time()` 硬对齐训练窗。  
- **Cue = mi_start**：Align 下同一时刻（`cue_s=0`）。  
- Rest 只认 `rest_start`–`rest_end`，不要把 prep 算进 Rest。

## 动手题

1. 打开 M00 会话的 `events.jsonl`，数一种事件出现次数。  
2. 若有 `alignment/verify_report.json`，读 `passed` 与各 check。  
3. 对照范式文档，把一条 Left 试次的事件序列抄在纸上。

## 验收

- [ ] 能说出至少 5 个 Align 事件名及其含义  
- [ ] 知道 verify_report 的用途  

## 下一模块

[`M04_试次状态机Align.md`](M04_试次状态机Align.md)
