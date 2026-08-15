# 14 · Shallow 前置 SE / ECA（旁路）

| 项 | 路径 |
|----|------|
| **完整方案** | [`方案.md`](./方案.md) |
| 结果登记 | [`总结/结果登记表.md`](./总结/结果登记表.md) |
| 代码 | `code/train_lab/src/step/5060_shallow_se_eca_accpaper/` |
| out | `code/train_lab/out/5060_shallow_se_eca_accpaper/` |

### 双臂（冻结）

| 臂 | 结构 |
|----|------|
| **A** | **SE → ShallowFBCSPNet**（输入 8 导通道重加权） |
| **B** | **ECA → ShallowFBCSPNet**（同上，无降维 1D 通道交互） |

协议：OpenBMI 2s/hop100 · 被试独立 · Acc_paper · balbatch · batch 128/256。  
对照：同协议原版 Shallow / 正式 0.694·0.540（只读）。  
禁止写入正式表；不与方案 13 CIACNet 绝对值夺冠混比。
