# 5060 · LeJEPA · Three 探测（旁路）

> 方案：`资料/模型训练/11_旁路_LeJEPA_Three探测_openbmi_accpaper/方案.md`  
> 相对 10 号：去掉 EMA，改为 **预测 + λ·SIGReg**。  
> **方案 B**（与 10 同构）：8ch×20=160 · 四块≈25%。

## 臂

| 脚本 | 含义 |
|------|------|
| `run_j1_pretrain.py` | LeJEPA 预训 → `j1/fold*_lejepa.pt` |
| `run_three_downstream.py --arm j0\|j2\|j2_random\|j3` | 同 10 号对照协议 |

## 冒烟（对齐 10 号加长设定）

```powershell
cd D:\cyy\MI\code\train_lab\src\step\5060_lejepa_three_probe_openbmi_accpaper
$env:PYTHONUNBUFFERED="1"

python -W ignore run_j1_pretrain.py --max-folds 1 --pretrain-epochs 50 --num-workers 0 --max-windows 50000 --batch 64
# 记下 run_... 路径为 $RUN

python -W ignore run_three_downstream.py --arm j2_random --max-folds 1 --max-epochs 30 --patience 10 --num-workers 0 --max-train-windows 20000 --resume-dir $RUN
python -W ignore run_three_downstream.py --arm j2 --j1-dir "$RUN\j1" --max-folds 1 --max-epochs 30 --patience 10 --num-workers 0 --max-train-windows 20000 --resume-dir $RUN
python -W ignore run_three_downstream.py --arm j0 --max-folds 1 --max-epochs 30 --patience 10 --num-workers 0 --max-train-windows 20000 --resume-dir $RUN
python -W ignore run_three_downstream.py --arm j3 --j1-dir "$RUN\j1" --max-folds 1 --max-epochs 30 --patience 10 --num-workers 0 --max-train-windows 20000 --resume-dir $RUN
```

成功线：J2−random≥+3pp 且 J3≥J0。
