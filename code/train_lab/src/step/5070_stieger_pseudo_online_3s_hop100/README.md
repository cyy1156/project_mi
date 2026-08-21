# 伪在线 08 · Stieger · shallow 3s · **RTX 5070**

方案：`资料/伪在线实验/08_旁路_OpenBMI_3s滑窗_Stieger零样本_5070/方案.md`  
协议同 [07（5060）](../../../资料/伪在线实验/07_旁路_OpenBMI_3s滑窗_Stieger零样本/)；**机位/权重/产物独立**。

## 前置

1. 离线 S3 权重（5070）：

```powershell
cd D:\cyy\MI\code\train_lab\src\step\5070_baselines_openbmi_3s_hop100_accpaper
D:\cyy\MI\.venv\Scripts\python.exe baseline_shallow.py --max-folds 0 --num-workers 2
```

2. Stieger 数据（与 07 共用）：

```powershell
cd D:\cyy\MI\code\preprocess_lab
D:\cyy\MI\.venv\Scripts\python.exe -m src.datasets.stieger.batch_3s_hop100
```

## 跑序

```powershell
cd D:\cyy\MI\code\train_lab\src\step\5070_stieger_pseudo_online_3s_hop100
$PY = "D:\cyy\MI\.venv\Scripts\python.exe"

$PY eval_zeroshot.py                          # S08-01 / Q0
$PY eval_gated.py --mode zeroshot             # S08-03 / Q1
$PY write_q0q1.py                             # S08-04
$PY ft_half.py                                # S08-02 ≡ S08-05
$PY eval_gated.py --mode ft                   # S08-06
```

烟雾：加 `--smoke`。

## 产物

写入 `资料/伪在线实验/08_旁路_OpenBMI_3s滑窗_Stieger零样本_5070/results/`（**不写** 07 / 01–06）。

| 目录 | 脚本 |
|------|------|
| `S08-01_zeroshot/` | `eval_zeroshot.py` |
| `S08-02_ft_half/` · `S08-05_ft_half/` | `ft_half.py` |
| `S08-03_gate/` | `eval_gated.py --mode zeroshot` |
| `S08-04_q0q1_table/` | `write_q0q1.py` |
| `S08-06_ft_gated/` | `eval_gated.py --mode ft` |

权重：`out/5070_baseline_openbmi_3s_hop100_accpaper/`  
FT：`out/5070_stieger_ft_3s_hop100_accpaper/`

## 禁止

- 覆盖 07 / `out/5060_*` / 游戏 01–06 `results/`
- 用 5060 的 `run_20260821_190504` 冒充本臂正式数（可作对照，须标注机位）
