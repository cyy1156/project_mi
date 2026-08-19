# 18 · 手写 ShallowFBCSP vs braindecode 库模型（旁路）

> **完整计划**：[`方案.md`](./方案.md)

| 项 | 说明 |
|----|------|
| 目标 | 同一 OpenBMI 2s/hop100 Acc_paper 滑窗环下，对比手写 vs 库 Shallow |
| 训练锚点 | `code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper/` |
| 手写实现 | `self_model/shallowfbcsp.py`（`attn=None`） |
| 库实现 | `braindecode.models.ShallowFBCSPNet` |
| 代码 | `code/train_lab/src/step/5070_shallow_impl_audit_accpaper/` |
| 状态 | **代码已就绪** · 待跑 A/B + 五折 |
