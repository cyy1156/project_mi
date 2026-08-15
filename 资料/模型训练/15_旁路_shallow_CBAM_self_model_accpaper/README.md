# 15 · self_model Shallow + CBAM（旁路）

| 项 | 路径 |
|----|------|
| **完整方案** | [`方案.md`](./方案.md) |
| 结果登记 | [`总结/结果登记表.md`](./总结/结果登记表.md) |
| 基座 | [`self_model/shallowfbcsp.py`](../../../self_model/shallowfbcsp.py) |
| 计划训练 | `self_model/train_shallow_cbam_hop100_accpaper.py` |

### 臂角色（冻结）

| 臂 | 结构 | 角色 |
|----|------|------|
| **A2** | **ConvTime → 仅通道 → ConvSpat → 仅时间空间** | **主推**（导联只归 ConvSpat） |
| **A1** | **ConvTime → 全 CBAM → ConvSpat…** | **对照**（测导联是否双重选择有害） |

S0 = 原版 self_model Shallow。成败以 **A2 vs S0** 为准。  
协议：OpenBMI 2s/hop100 · Acc_paper。不做输入端前置（方案 14 阴性）。
