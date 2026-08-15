# 11 · LeJEPA Three 探测（旁路）

| 项 | 路径 |
|----|------|
| **完整方案** | [`方案.md`](./方案.md) |
| 结果登记 | [`总结/结果登记表.md`](./总结/结果登记表.md) |
| 训练代码 | `code/train_lab/src/step/5060_lejepa_three_probe_openbmi_accpaper/` |
| 10 号对照 | [`../10_旁路_JEPA最小原型_Three探测_openbmi_accpaper/`](../10_旁路_JEPA最小原型_Three探测_openbmi_accpaper/) |
| 12 号（BCI2a·4s） | [`../12_旁路_LeJEPA_Three探测_bci2a_4s_accpaper/`](../12_旁路_LeJEPA_Three探测_bci2a_4s_accpaper/) |

要点：`L = L_pred + λ·SIGReg`，无 EMA；与 10 号同 **方案 B** token/mask（8ch×100ms、四块≈25%），只换预训目标。成功线见方案 §0.2。
