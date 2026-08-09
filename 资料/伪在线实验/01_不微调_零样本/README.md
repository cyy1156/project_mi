# 01 · 不微调 / 零样本

本臂：**不对游戏被试微调或重训**；只读加载 BCI2a `balbatch_accpaper` 权重做伪在线推理。  
**数字已冻结**，作 BCI2a→游戏对照。

> 部署口径（2026-08-09 起）：零样本主权重改为 **OpenBMI 正式 shallow**，见 [04](../04_旁路_OpenBMI权重_游戏零样本与门控/) 与 [上级「当前部署口径」](../README.md)。本臂结果不改写。

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 协议全文 |
| [`实验结果汇总.md`](./实验结果汇总.md) | 正式五折汇总（已冻结） |
| `out/` | 切段索引 |
| `results/` | 各模型正式 run |

上级索引：[../README.md](../README.md)  
代码：`code/train_lab/src/step/game_pseudo_online_hop100/`（`DOCS_OUT` → 本目录）
