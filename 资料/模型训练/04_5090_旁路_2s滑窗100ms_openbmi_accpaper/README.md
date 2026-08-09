# 04_5090 · OpenBMI · 2s/hop100 · Acc_paper 选模（RTX 5090 · **对照 · 非正式**）

> 状态：**十一模型方案** · 2026-08-05  
> **非正式对照**；正式 Acc_paper 以 [`../5060_openbmi_accpaper_实验与权重清单.md`](../5060_openbmi_accpaper_实验与权重清单.md) 为准。  
> 双机总览：[`../OpenBMI_Acc_paper_双机目录.md`](../OpenBMI_Acc_paper_双机目录.md)

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 协议说明（5090 侧） |
| 训练（5090 · 对照） | `code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper/` |
| 训练（5060 · 正式） | `code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper/` |
| 五折记录 | `资料/模型训练/runs/5090_openbmi_accpaper/` |
| 对照清单 | [`../5090_openbmi_accpaper_实验与权重清单.md`](../5090_openbmi_accpaper_实验与权重清单.md) |

```bash
cd code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper
python run_all.py --continue-on-error
```
