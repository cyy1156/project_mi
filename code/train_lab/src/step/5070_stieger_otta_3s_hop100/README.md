# 09 · OTTA（EA + AdaBN）· Stieger 3s 伪在线回放

在 [07](../../../../../资料/伪在线实验/07_旁路_OpenBMI_3s滑窗_Stieger零样本/) 协议上**只改推理侧**：叠加 EA + AdaBN，可选 H2 门控；C1 检验在线伪标签全量 FT。

| 项 | 值 |
|----|-----|
| 包路径 | `code/train_lab/src/step/5070_stieger_otta_3s_hop100` |
| 零样本权重 | 5060 S3 `run_20260821_190504`（与 07 同源） |
| FT 权重 | `stieger_ft_3s_hop100_accpaper/.../20260822_153300` |
| 数据 | `preprocess_lab/out/stieger_3s_hop100`（须含 `stieger_X_noz.npy`） |
| 结果 | `资料/伪在线实验/09_.../results/S09-*` |

## 快速开始

```powershell
cd D:\MI\code\train_lab\src\step\5070_stieger_otta_3s_hop100
C:\Users\yy\.conda\envs\cyy\python.exe _smoke_local.py   # 单被试冒烟
.\run_all_09.ps1                                          # 主线 A3+B3+B4+C1
```

## 单臂评测

```powershell
# 零样本完整 OTTA
python eval_ab.py --arm A3

# FT + OTTA + H2（部署候选）
python eval_ab.py --arm B4

# 多臂
python eval_ab.py --arms A1,A2,A3b,A4,B1,B2,B5

# C1 伪标签在线 FT（基于 B3）
python eval_c1.py
```

## EA ref=src

首次运行会从 `DATA/openbmi/sess*_subj*_EEG_MI.mat` 在线估计参考协方差并缓存到：

`train_lab/out/stieger_otta_3s_hop100_accpaper/ea_ref_src_cov.npy`

正式跑前可选生成 noz 语料（更规范）：

```powershell
cd D:\MI\code\preprocess_lab
python -m src.datasets.openbmi.batch_3s_hop100 --no-zscore
```

## 臂一览

| 臂 | 权重 | EA | AdaBN | 门控 |
|----|------|-----|-------|------|
| A0/B0 | 零样本/FT | — | — | H0 |
| A3/B3 | 零样本/FT | src | ✓ | H0 |
| B4 | FT | src | ✓ | H2 |
| C1 | FT（B3+伪标签 FT） | src | ✓ | — |

评测均为**后半 trial**因果流式；锚点 A0/B0 数值见 `资料/.../09/总结/结果登记表.md`。

## 模块

- `ea.py` — 欧氏对齐（前半估计 R_s → ref）
- `adabn.py` — 仅更新 BN running 统计
- `ref_cov.py` — OpenBMI 源域参考协方差
- `eval_ab.py` — A/B 系列
- `eval_c1.py` — C1 伪标签全量 FT

复用 07 包：`data` / `data_split` / `infer` / `weights` / `util_metrics`。
