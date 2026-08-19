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

主链五折已于 **2026-08-19** 跑完（22 臂）。详见 [`结果登记表.md`](结果登记表.md) 与 `code/train_lab/out/5090_mask_future_dual_expert_accpaper/_scheme17_summary_table.json`。

| 臂 | Test Acc_paper mean±std | 备注 |
|----|-------------------------|------|
| A0_ref/Task | 0.6909±0.038 | braindecode 参考 |
| A0_ref/Three | 0.5425±0.031 | braindecode 参考 |
| A1 | 0.5754±0.021 | pf1000 单专家基线 |
| P2（主） | 0.5703±0.022 | 定稿主结果 |
| B9 | 0.5788±0.019 | leak oracle 上界 |
| C2a | 0.5758±0.020 | 最佳 C* |

U 系列附报已于 **2026-08-19** 全量完成；最佳 **U13 = 0.5753±0.022**（≈ A1）。详见 [`结果登记表.md`](结果登记表.md) § U 系列。

权重与 metrics 根目录：`code/train_lab/out/5090_mask_future_dual_expert_accpaper/`
