# M01 · 串口 → BrainFlow → LSL（你的舒适区）

## 目标

搞清：**你熟悉的 COM 在本项目里接到哪一层结束**；之后为何改谈 LSL。

## 关键概念

```text
你熟悉的世界          本项目封装               实验侧只认这个
─────────────        ──────────────           ────────────
COM 口 / Cyton   →   BrainFlow 驱动      →   LSL 流 "OpenBCI_EEG"
                     (+ 可选板端滤波)         @ 250 Hz
```

真机：必须先关 OpenBCI GUI，避免占 COM。  
联调：合成板不占真实串口（`use_synthetic=True`）。

## 对应代码

| 内容 | 路径 | 看什么 |
|------|------|--------|
| 采集门面 | `acquisition/service.py` | 类 `AcquisitionFacade`：`__init__`、`create`、`start`、`stop`、`preflight_probe` |
| 路径注入 | 同文件 `LSL_CONNECT_ROOT`、`ensure_lsl_connect_on_path` | 指向 `collect_data/LSL_connect_model/...` |
| COM 枚举 | `experiment/serial_ports.py` | 操作台如何列端口 |
| 通道默认 | `core/channel_layout.py` | `DEVICE_CHANNEL_LABELS`（8 导冻结序） |
| 底层实现 | `collect_data/LSL_connect_model/LSL_connect_model/` | ServiceManager / preprocessing（可略读） |

注意：`AcquisitionFacade` 默认 **`filter_enabled=False`**（避免与模型侧滤波双重滤波）；滤波主责在在线预处理。

## 精读顺序（建议 1–2 小时）

1. `acquisition/service.py` 全文骨架（该类不长）  
2. 搜索仓库谁调用了 `AcquisitionFacade`（多半在 `orchestrator` 开采集时）  
3. 对照 `core/channel_layout.py` 通道列表与帽上丝印

## 动手题

1. 画一张图：COM → Facade.start → LSL 流名。  
2. 真机时：用 `preflight_probe` 的语义解释「会话前 2s 探针」在防什么。  
3. （可选）合成板再跑 1 分钟，确认不依赖真实 COM。

## 验收

- [ ] 能指出「串口逻辑在 `collect_data/...`，experiment_game 只做门面」  
- [ ] 能说出 EEG LSL 流默认名与采样率  

## 下一模块

[`M02_RingBuffer与EEGBus.md`](M02_RingBuffer与EEGBus.md)
