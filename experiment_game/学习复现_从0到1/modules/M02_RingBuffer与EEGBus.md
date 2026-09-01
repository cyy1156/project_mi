# M02 · RingBuffer + EEGBus + 落盘

## 目标

理解：**为什么不能「落盘一路、推理一路各拉一股 LSL」**；总线 + 多订阅者怎么接。

## 关键概念

历史上若两路各自 `LSL inlet`，一路断了另一路还在用旧缓冲 → 会出现「模型假塌缩」（恒定概率）。  
现行设计：**只拉一次流进 RingBuffer，经 EEGBus 扇出**。

```text
LSL inlet ──► RingBuffer.push ──► EEGBus.publish
                                    ├─ CsvRecorderSubscriber → eeg.csv
                                    ├─ InferenceService（读同一 buf）
                                    ├─ 前端波形
                                    └─ 健康 / watchdog
```

## 对应代码

| 内容 | 路径 | 看什么 |
|------|------|--------|
| 环形缓冲 + LSL 附着 | `experiment/inference_v2.py` | 类 `RingBuffer`：`attach_lsl`、`push`、取窗相关方法 |
| 总线 | `runtime/eeg_bus.py` | publish / subscribe |
| 真机捕获组装 | `experiment/live_capture.py` | `LiveEegCapture`：把 buf、bus、csv 接起来 |
| CSV 订阅者 | `runtime/csv_recorder.py` | 如何写 `eeg.csv` |
| 健康 | `runtime/eeg_health.py` | stall / stale 概念（可略） |

## 精读顺序

1. `live_capture.py`（短，看组装）  
2. `runtime/eeg_bus.py`  
3. `inference_v2.py` 里 **只读 `RingBuffer` 类**（先跳过 InferenceService）  
4. `csv_recorder.py` 写文件部分  

## 动手题

1. 在纸上列出 4 个订阅者。  
2. 打开一场 M00 的 `eeg.csv` 前 20 行，对照通道列是否像 `FC3…CP4`。  
3. （进阶）在 `RingBuffer.push` 临时打日志：确认合成板有持续样本；验证后删掉。

## 验收

- [ ] 能解释 Bus 相对「双路拉流」的好处  
- [ ] 能指出 CSV 写入类名与调用链  

## 下一模块

[`M03_Marker与时间对齐.md`](M03_Marker与时间对齐.md)
