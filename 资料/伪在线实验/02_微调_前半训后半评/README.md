# 02 · 微调（前半训 / 后半评）

> 状态：**方案已确认 · 代码已落地** · 2026-08-05  
> 微调范围：**全模型权重**（`finetune_mode=full_model`；禁止只训分类头）

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 协议全文 |
| `out/` | `split_manifest.json` + 流产物 |
| `results/` | 各模型正式 / 冒烟 run |

代码：`code/train_lab/src/step/game_ft_hop100_accpaper/`  
对照臂（冻结）：[`../01_不微调_零样本/`](../01_不微调_零样本/)
