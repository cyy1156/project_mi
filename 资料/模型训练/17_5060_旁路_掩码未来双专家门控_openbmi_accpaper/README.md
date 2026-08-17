# 17 · 5060 旁路 · 掩码未来双专家门控 · OpenBMI Acc_paper

> 方案正文：[`资料/模型方案/掩码未来表征预测_双专家门控_在线MI/`](../../模型方案/掩码未来表征预测_双专家门控_在线MI/)  
> 代码包：[`code/train_lab/src/step/5060_mask_future_dual_expert_accpaper/`](../../../code/train_lab/src/step/5060_mask_future_dual_expert_accpaper/)  
> **全量五折请用** [`../17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/`](../17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/)

## 设备

NVIDIA RTX 5060 Laptop · ~16GB RAM · 低内存默认（batch 64/128 · workers=0 · 默认 fold0）

## 一键启动

```powershell
cd code/train_lab/src/step/5060_mask_future_dual_expert_accpaper
python _smoke_local.py
python chain_all.py
# 或 run_chain_detached.bat
```

A1+ 前置数据：

```powershell
cd code/preprocess_lab
python -m src.datasets.openbmi_pf1000.batch
# 输出 out/openbmi_2s_hop100_pf1000/
```

默认链：`A0_ref → A0 → A1 → P0 → A2 → P1 → P2`（每臂 fold0）。  
完整 B/C：`python chain_all.py --full-chain`。

## 结果登记

| 臂 | Test Acc_paper（fold0 / 五折） | run 路径 | 备注 |
|----|-------------------------------|----------|------|
| A0 | | | |
| A1 | | | |
| P0 | | | |
| A2 | | | |
| P1 | | | |
| P2（主） | | | |

权重根目录：`code/train_lab/out/5060_mask_future_dual_expert_accpaper/`
