# 伪在线 07 · Stieger · shallow 3s

方案：`资料/伪在线实验/07_旁路_OpenBMI_3s滑窗_Stieger零样本/方案.md`

## 前置

```powershell
# 1) 确保 DATA/stieger 有 S*_Session_*.mat
cd D:\cyy\MI\code\preprocess_lab
D:\cyy\MI\.venv\Scripts\python.exe -m src.datasets.stieger.batch_3s_hop100
# 产物：out/stieger_3s_hop100/stieger_X.npy + stieger_X_noz.npy + ...
```

权重默认：`run_20260821_190504`（OpenBMI Acc_paper S3 shallow）。

## 跑序（在包目录内，与其它伪在线臂一致）

```powershell
cd D:\cyy\MI\code\train_lab\src\step\stieger_pseudo_online_3s_hop100
$PY = "D:\cyy\MI\.venv\Scripts\python.exe"

# A / S07-01 全量零样本
$PY eval_zeroshot.py

# B / S07-03 零样本 + H0–H3
$PY eval_gated.py --mode zeroshot

# S07-04 读数表（链 01+03）
$PY write_q0q1.py

# C / S07-02(=05) 前半 FT
$PY ft_half.py

# D / S07-06 FT 后半 + 门控
$PY eval_gated.py --mode ft
```

烟雾：各脚本加 `--smoke`（仅 fold0；FT 再缩短 epoch）。

## 产物

全部写入 `资料/伪在线实验/07_旁路_OpenBMI_3s滑窗_Stieger零样本/results/`：

| 目录 | 脚本 |
|------|------|
| `S07-01_zeroshot/` | `eval_zeroshot.py` |
| `S07-02_ft_half/` · `S07-05_ft_half/` | `ft_half.py` |
| `S07-03_gate/` | `eval_gated.py --mode zeroshot` |
| `S07-04_q0q1_table/` | `write_q0q1.py` |
| `S07-06_ft_gated/` | `eval_gated.py --mode ft` |

FT 权重：`code/train_lab/out/stieger_ft_3s_hop100_accpaper/`。

## 注意

- 通道已是 OpenBMI 序，**无游戏重排**。
- 门控在 **`X_noz`** 上算 ERD/laterality；推理用 **`X`**（窗内 z-score）。
- Rest/Task 的 `trial_id` 成对（+0/+1）；门控与半程划分用 **`cue_id`**。
