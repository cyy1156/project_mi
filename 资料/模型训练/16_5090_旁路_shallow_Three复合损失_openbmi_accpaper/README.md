# 16 · 5090 · Shallow Three 复合损失（旁路 · 全量）

> **机位：RTX 5090 · 内存 128GB · 显存 32GB** · **全量五折推荐入口**。  
> 损失/臂定义与 [`../16_5060_旁路_shallow_Three复合损失_openbmi_accpaper/方案.md`](../16_5060_旁路_shallow_Three复合损失_openbmi_accpaper/方案.md) **相同**。  
> 本包结果默认记为 **对照**（非正式十一模型表）。

| 项 | 路径 |
|----|------|
| **完整方案（共享）** | [`../16_5060_旁路_shallow_Three复合损失_openbmi_accpaper/方案.md`](../16_5060_旁路_shallow_Three复合损失_openbmi_accpaper/方案.md) |
| 本机说明 | [`方案.md`](./方案.md) |
| 结果登记 | [`总结/结果登记表.md`](./总结/结果登记表.md) |
| 代码 | `code/train_lab/src/step/5090_three_hier_loss_accpaper/` |
| out | `code/train_lab/out/5090_three_hier_loss_accpaper/` |

## 一键全链（5090）

```powershell
conda activate cyy
cd F:\Cyy\MI\code\train_lab\src\step\5090_three_hier_loss_accpaper
python chain_all.py
# 或双击 run_chain_detached.bat（需先 conda activate cyy）
```

顺序：Three **S0→H1→H2→H3**（各五折）→ Task **T0**（五折）。
