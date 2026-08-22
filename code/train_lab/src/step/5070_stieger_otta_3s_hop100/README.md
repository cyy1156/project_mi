# 09 · OTTA（EA + AdaBN）· Stieger 3s 伪在线回放 · **v1.2**

在 [07](../../../../../资料/伪在线实验/07_旁路_OpenBMI_3s滑窗_Stieger零样本/) 协议上**只改推理侧**：统一 `X_noz` 管线、EA(cal) 白化、predict-first AdaBN；C1 检验在线伪标签全量 FT。

| 项 | 值 |
|----|-----|
| 包路径 | `code/train_lab/src/step/5070_stieger_otta_3s_hop100` |
| AdaBN | **v1.2**（eval 预测 → train 更新 running stats） |
| 输入 | `noz_unified`：全体 `X_noz` → (EA) → z-score |
| 零样本权重 | 5060 S3 `run_20260821_190504` |
| FT 权重 | `stieger_ft_3s_hop100_accpaper/.../20260822_153300` |
| 数据 | `preprocess_lab/out/stieger_3s_hop100`（须含 `stieger_X_noz.npy`） |
| 结果 | `资料/伪在线实验/09_.../results/S09-*` |

v1.0/v1.1 已作废；见 `PAUSED.md`。

## 快速开始

```powershell
cd D:\MI\code\train_lab\src\step\5070_stieger_otta_3s_hop100
C:\Users\yy\.conda\envs\cyy\python.exe archive_invalid_runs.py
C:\Users\yy\.conda\envs\cyy\python.exe _smoke_local.py   # S1 冒烟
.\run_all_09.ps1                                          # v1.2 全链
```

## 单臂评测

```powershell
python eval_ab.py --arm A3
python eval_ab.py --arm B4
python eval_ab.py --arms A0,A1,A2,A3,B0,B1,B2,B3,B4
python eval_c1.py   # 须先完成全量 A0 + B3（24 被试）
```

## EA 参考协方差

- **A 系 `src`**：OpenBMI 源域（`ea_ref_src_cov.npy`）
- **B 系 `cal`**：被试前半标定段白化（`R_ref = (trace/8)·I`）

## 臂一览

| 臂 | 权重 | EA | AdaBN | 门控 | 配对基线 |
|----|------|-----|-------|------|----------|
| A0/B0 | 零样本/FT | off | — | H0 | — |
| A1/A3 | 零样本 | src | —/✓ | H0 | vs A0 |
| B1/B3/B4 | FT | cal | —/✓/✓ | H0/H2 | vs B0 |
| C1 | FT（B3+伪标签 FT） | cal | ✓ | — | vs 同跑 A0/B3 |

主判定 Δ 用**同跑 v1.2 A0/B0**；S07 全量锚点只读。

## 模块

- `ea.py` / `adabn.py` / `ref_cov.py` / `eval_ab.py` / `eval_c1.py`
- `paired_results.py` — v1.2 配对查找 + C1 预检
- `archive_invalid_runs.py` — 归档无效 run

复用 07 包：`data` / `data_split` / `infer` / `weights` / `util_metrics`。
