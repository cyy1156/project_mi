# 12 · LeJEPA Three 探测（BCI2a · 固定 4s）

旁路代码包。方法继承方案 11（LeJEPA + 方案 B mask）；数据为 `bci2a_4s`（8×1000）。

| 项 | 路径 |
|----|------|
| 方案文档 | `资料/模型训练/12_旁路_LeJEPA_Three探测_bci2a_4s_accpaper/` |
| out | `code/train_lab/out/5060_lejepa_three_probe_bci2a_4s_accpaper/` |
| 母本 | `../5060_lejepa_three_probe_openbmi_accpaper/` |

## 相对方案 11

- `data_tag=bci2a_4s` · `n_times=1000` · token **320**
- 无 `trial_id.npy` → `np.arange(N)`
- batch：**128 / 128 / 256**
- 下游早停：**Val BalAcc**

## 冒烟

```powershell
cd D:\cyy\MI\code\train_lab\src\step\5060_lejepa_three_probe_bci2a_4s_accpaper
$env:PYTHONUNBUFFERED = "1"
python -W ignore run_j1_pretrain.py --max-folds 1 --pretrain-epochs 2 --num-workers 0 --batch 64
```

## S-长 fold0

```powershell
python -W ignore run_j1_pretrain.py --max-folds 1 --pretrain-epochs 50 --num-workers 2 --batch 128
# $RUN = ...\out\5060_lejepa_three_probe_bci2a_4s_accpaper\run_*
python -W ignore run_three_downstream.py --arm j2_random --max-folds 1 --max-epochs 30 --patience 10 --num-workers 2 --resume-dir $RUN
python -W ignore run_three_downstream.py --arm j2 --j1-dir "$RUN\j1" --max-folds 1 --max-epochs 30 --patience 10 --num-workers 2 --resume-dir $RUN
python -W ignore run_three_downstream.py --arm j0 --max-folds 1 --max-epochs 30 --patience 10 --num-workers 2 --resume-dir $RUN
python -W ignore run_three_downstream.py --arm j3 --j1-dir "$RUN\j1" --max-folds 1 --max-epochs 30 --patience 10 --num-workers 2 --resume-dir $RUN
```
