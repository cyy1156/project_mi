# 10 · 5060 统一包 · 3s 复现 + **2s 窗长** + OTTA v1.2

| 项 | 值 |
|----|-----|
| 包路径 | `code/train_lab/src/step/5060_stieger_unified_3s_hop100` |
| 方案 | `资料/伪在线实验/10_.../方案.md` |
| 3s 权重 | `run_20260821_190504` |
| **2s 权重** | `run_20260807_135828`（须同步至本机） |
| OTTA | v1.2 严格对齐（`noz_unified` + EA 白化 + predict_first） |

## 预处理（一次）

```powershell
cd D:\MI\code\preprocess_lab
python -m src.datasets.stieger.batch_3s_hop100
python -m src.datasets.stieger.batch_2s_hop100   # Part II 必需
```

## 冒烟

```powershell
cd D:\MI\code\train_lab\src\step\5060_stieger_unified_3s_hop100
C:\Users\yy\.conda\envs\cyy\python.exe _smoke_local.py
```

## Part II · 2s 窗长对照（主线）

```powershell
.\run_part2_2s.ps1
# 或单步：
python eval_zeroshot.py --tw 2s    # S10-01b
python ft_half.py --tw 2s          # S10-02b
```

## Part III · OTTA（仅 3s）

```powershell
python eval_otta.py --arms A0,A1,A2,A3,B0,B1,B2,B3,B4
python eval_c1.py
```

## `--tw` 联动项

| 参数 | 3s | 2s |
|------|----|----|
| `n_times` | 750 | 500 |
| 数据 | `stieger_3s_hop100` | `stieger_2s_hop100` |
| 权重包 | `openbmi_3s_hop100` | `openbmi_2s_hop100` |
| 零样本臂 | S10-01 | **S10-01b** |
| FT 臂 | S10-02 | **S10-02b** |

复用 07 包：`data_split` / `infer` / `util_metrics` / `dataset` / `task_sampler`。
