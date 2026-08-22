# 09 · shallow · 3s · OTTA 测试时自适应（EA + AdaBN + 置信度滤波）· Stieger 回放

> 状态：**方案已立 · 待跑** · 2026-08-22  
> 详案：[方案.md](方案.md)  
> 登记：[总结/结果登记表.md](总结/结果登记表.md)（占位）

在 [07（Stieger 3s 复现）](../07_旁路_OpenBMI_3s滑窗_Stieger零样本/) 的协议与权重上，**只改推理侧**：叠加论文《Fine-Tuning Strategies for Continual Online EEG MI Decoding》第二分支 **OTTA**（EA 欧氏对齐 + AdaBN，全程冻结判别权重），并设 **C 系列**检验「小模型在线全量微调」是否优于冻结 OTTA。

| 要点 | 说明 |
|------|------|
| 权重锚点 | 零样本臂 = 5060 S3 `run_20260821_190504`；FT 臂 = S07-02/05 ckpt `20260822_153300`（**复用，不重训**） |
| 评测 | Stieger `stieger_3s_hop100` · 24 被试 · 后半 trial · 因果流式（前半估计对齐量，后半回放） |
| 成本 | A/B 系列**纯回放零训练**；C 系列在线更新亦为轻量（shallow ~17k 参数） |
| 预注册主线 | B4（FT + EA + AdaBN + H2 门控）目标 Three ≥ 0.68；OTTA 净效应 B3−B0 ≥ +1.5 pp |

**禁止**：覆盖 07/08 的 out 与 results；与游戏 04 表混比夺冠；在线阶段读取未来窗或标签。
