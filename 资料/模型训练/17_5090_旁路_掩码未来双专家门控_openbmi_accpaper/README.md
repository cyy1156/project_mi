# 17 · 5090 旁路 · 掩码未来双专家门控 · OpenBMI Acc_paper

> 方案正文：[`资料/模型方案/掩码未来表征预测_双专家门控_在线MI/`](../../模型方案/掩码未来表征预测_双专家门控_在线MI/)  
> 代码包：[`code/train_lab/src/step/5090_mask_future_dual_expert_accpaper/`](../../../code/train_lab/src/step/5090_mask_future_dual_expert_accpaper/)  
> **本机低内存旁路**：[`../17_5060_旁路_掩码未来双专家门控_openbmi_accpaper/`](../17_5060_旁路_掩码未来双专家门控_openbmi_accpaper/) · `5060_mask_future_dual_expert_accpaper/`

## 设备

NVIDIA RTX 5090 · RAM 128GB · VRAM 32GB（与方案 16 旁路同机 · `F:\Cyy\MI` · conda `cyy`）

### 本机适配要点

| 项 | 状态 |
|----|------|
| A0 / 自写臂（500pt） | ✅ 需已有 `out/openbmi_2s_hop100` |
| A0_ref（braindecode） | ✅ Windows 默认 `num_workers=0`（可显式 `--num-workers 2` 试跑） |
| A1+（1000pt pf1000） | ⚠️ 需三类数据（Rest+L/R，`protocol_version≥3`）；旧 no_rest 需 `--reset` 重跑 |
| 原始 `.mat` | 本机路径 `F:\Cyy\MI\DATA\openbmi\openbmi\sess*_subj*_EEG_MI.mat`（**两层** openbmi，非三层） |
| 与其他 GPU 任务并存 | 建议 `--batch-train 128 --num-workers 2` |

## 一键启动

```powershell
cd code/train_lab/src/step/5090_mask_future_dual_expert_accpaper
python chain_all.py
# 或 run_chain_detached.bat
```

A1+ 前置数据：

```powershell
cd F:\Cyy\MI\code\preprocess_lab   # 或 D:\cyy\MI\code\preprocess_lab
python -m src.datasets.openbmi_pf1000.batch --reset
# 输出 out/openbmi_2s_hop100_pf1000/ · 三类 y_three∈{0,1,2}
# Left/Right：从 cue 起切；Rest：Cue 前满 5.6s 同几何
# 若 shard 已齐仅缺合并：python -m src.datasets.openbmi_pf1000.batch --merge-only
```

原始 `.mat` 路径：`F:\Cyy\MI\DATA\openbmi\openbmi\sess*_subj*_EEG_MI.mat`（**两层** openbmi，不是三层 `openbmi/openbmi/openbmi`）。

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
