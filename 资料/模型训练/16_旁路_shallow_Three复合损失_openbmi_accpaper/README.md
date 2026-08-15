# 16 · Shallow Three 复合损失（旁路）

| 项 | 路径 |
|----|------|
| **完整方案** | [`方案.md`](./方案.md) |
| 结果登记 | [`总结/结果登记表.md`](./总结/结果登记表.md) |
| 代码 | `code/train_lab/src/step/5060_three_hier_loss_accpaper/` |
| out | `code/train_lab/out/5060_three_hier_loss_accpaper/` |

**骨干冻结：braindecode ShallowFBCSPNet（小参数 ~1.6e4）**。  
目标冲刺：Task **0.75** / Three **0.60**（靠损失，不加大模型）。  
禁止写入正式表；不与方案 15 CBAM 混跑同一 out。
