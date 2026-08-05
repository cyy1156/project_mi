# 04 · OpenBMI · 2s/hop100 · Acc_paper 选模

> 状态：**方案已确认 · 代码已落地** · 2026-08-05  
> 被试键：**A** · patience=20 · Task+Three · Top-8

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 协议 + **§1.6 预处理步骤** |
| 预处理 | `code/preprocess_lab/src/datasets/openbmi/` |
| 训练 | `code/train_lab/src/step/baselines_openbmi_2s_hop100_accpaper/` |

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --limit 1 --reset
```
