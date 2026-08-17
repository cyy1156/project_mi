# 17 · 5090 旁路 · 掩码未来双专家门控 · OpenBMI Acc_paper

> 方案正文：[`资料/模型方案/掩码未来表征预测_双专家门控_在线MI/`](../../模型方案/掩码未来表征预测_双专家门控_在线MI/)  
> 代码包：[`code/train_lab/src/step/5090_mask_future_dual_expert_accpaper/`](../../../code/train_lab/src/step/5090_mask_future_dual_expert_accpaper/)  
> **本机低内存旁路**：[`../17_5060_旁路_掩码未来双专家门控_openbmi_accpaper/`](../17_5060_旁路_掩码未来双专家门控_openbmi_accpaper/) · `5060_mask_future_dual_expert_accpaper/`

## 设备

NVIDIA RTX 5090 · RAM 128GB · VRAM 32GB（与方案 16 旁路同机）

## 一键启动

```powershell
cd code/train_lab/src/step/5090_mask_future_dual_expert_accpaper
python chain_all.py
# 或 run_chain_detached.bat
```

A1+ 前置数据（本机有 mat 时）：

```powershell
cd code/preprocess_lab
python -m src.datasets.openbmi_pf1000.batch
# 输出 out/openbmi_2s_hop100_pf1000/
```

臂顺序：`A0_ref`（braindecode 参考）→ `A0`（自写主表）→ `A1` → … → `P2`。

## 结果登记

| 臂 | Test Acc_paper mean±std | run 路径 | 备注 |
|----|-------------------------|----------|------|
| A0 | | | |
| A1 | | | |
| P0 | | | |
| A2 | | | |
| P1 | | | |
| P2（主） | | | |
| … | | | B/C 见各 summary.json |

权重根目录：`code/train_lab/out/5090_mask_future_dual_expert_accpaper/`
