# 09 · OTTA（EA + AdaBN）· Stieger 3s 伪在线回放 · **v1.1**

在 [07](../../../../../资料/伪在线实验/07_旁路_OpenBMI_3s滑窗_Stieger零样本/) 协议上**只改推理侧**：叠加 EA + AdaBN，可选 H2 门控；C1 检验在线伪标签全量 FT。

| 项 | 值 |
|----|-----|
| 包路径 | `code/train_lab/src/step/5070_stieger_otta_3s_hop100` |
| AdaBN | **v1.1**（train 更新 running stats → eval 预测） |
| 零样本权重 | 5060 S3 `run_20260821_190504`（与 07 同源） |
| FT 权重 | `stieger_ft_3s_hop100_accpaper/.../20260822_153300` |
| 数据 | `preprocess_lab/out/stieger_3s_hop100`（须含 `stieger_X_noz.npy`） |
| 结果 | `资料/伪在线实验/09_.../results/S09-*` |

**暂停说明**：v1.0 因 AdaBN bug 已作废，见 `PAUSED.md`。v1.1 修复后重跑。

## 快速开始

```powershell
cd D:\MI\code\train_lab\src\step\5070_stieger_otta_3s_hop100
C:\Users\yy\.conda\envs\cyy\python.exe _smoke_local.py   # S1 冒烟（A0/A2/A3/B0/B2/B3 + C1）
.\run_all_09.ps1                                          # v1.1 全链：A0–A3 → B0–B4 → C1
```

## 单臂评测

```powershell
# 零样本完整 OTTA
python eval_ab.py --arm A3

# FT + OTTA + H2（部署候选）
python eval_ab.py --arm B4

# 多臂
python eval_ab.py --arms A0,A1,A2,A3,B0,B1,B2,B3,B4

# C1 伪标签在线 FT（基于 B3 v1.1，EA=cal）
python eval_c1.py
```

## EA 参考协方差

- **A 系**：`ref=src`（OpenBMI 源域，零样本）
- **B 系**：`ref=cal`（被试前半标定段，与 FT 对齐）

首次 A 系运行会从 `DATA/openbmi/sess*_subj*_EEG_MI.mat` 估计参考协方差并缓存到：

`train_lab/out/stieger_otta_3s_hop100_accpaper/ea_ref_src_cov.npy`

## 臂一览

| 臂 | 权重 | EA | AdaBN | 门控 | 配对基线 |
|----|------|-----|-------|------|----------|
| A0/B0 | 零样本/FT | — | — | H0 | — |
| A1/A3 | 零样本 | src | —/✓ | H0 | vs A0 |
| B1/B3/B4 | FT | **cal** | —/✓/✓ | H0/H2 | vs B0 |
| C1 | FT（B3+伪标签 FT） | cal | ✓ | — | vs B3 |

评测均为**后半 trial**因果流式；主判定 Δ 用**同跑 A0/B0**；S07 全量锚点只读参考。

## 模块

- `ea.py` — 欧氏对齐（前半估计 R_s → ref）
- `adabn.py` — v1.1 流式 AdaBN（仅更新 BN running 统计）
- `ref_cov.py` — OpenBMI 源域参考协方差
- `eval_ab.py` — A/B 系列
- `eval_c1.py` — C1 伪标签全量 FT

复用 07 包：`data` / `data_split` / `infer` / `weights` / `util_metrics`。
