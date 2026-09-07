# 5060 · JEPA 最小原型 · Three 探测（旁路）

> 方案：`资料/模型训练/10_旁路_JEPA最小原型_Three探测_openbmi_accpaper/方案.md`  
> **非正式表**；不覆盖正式 shallow 权重。  
> **方案 B**：8ch×20 token（160）· 四块并集≈25% · 禁止同 t 灭满 8 通道。

## 臂

| 命令 | 含义 |
|------|------|
| `run_j1_pretrain.py` | 纯 \(L_{jepa}\) 预训（排除 test 被试） |
| `run_three_downstream.py --arm j0` | 同骨干从头监督 Three |
| `run_three_downstream.py --arm j2_random` | 随机 Enc 冻住 + 线性头 |
| `run_three_downstream.py --arm j2 --j1-dir .../j1` | JEPA 冻住探测 |
| `run_three_downstream.py --arm j3 --j1-dir .../j1` | JEPA 解冻微调 |

## 冒烟（已跑通示例）

```powershell
cd D:\cyy\MI\code\train_lab\src\step\5060_jepa_three_probe_openbmi_accpaper
$env:PYTHONUNBUFFERED="1"

python -W ignore run_j1_pretrain.py --max-folds 1 --pretrain-epochs 5 --num-workers 0 --max-windows 8192 --batch 64
$RUN = "D:\cyy\MI\code\train_lab\out\5060_jepa_three_probe_openbmi_accpaper\run_20260811_110925"

python -W ignore run_three_downstream.py --arm j2_random --max-folds 1 --max-epochs 25 --patience 8 --num-workers 0 --max-train-windows 8192 --resume-dir $RUN
python -W ignore run_three_downstream.py --arm j2 --j1-dir "$RUN\j1" --max-folds 1 --max-epochs 25 --patience 8 --num-workers 0 --max-train-windows 8192 --resume-dir $RUN
```

冒烟结果（fold0）：J2 JEPA **0.325** vs 随机 **0.319**（Δ≈+0.5pp，未达 +3pp 线）。登记见方案目录 `总结/结果登记表.md`。

加长预训后再比（建议下一步）：

```powershell
python -W ignore run_j1_pretrain.py --max-folds 1 --pretrain-epochs 50 --num-workers 0 --max-windows 50000 --batch 64
```

全量五折：去掉 `--max-folds` / `--max-windows`，J1 默认 50 epoch；J2/J3 默认 300/20。

## 输出

`code/train_lab/out/5060_jepa_three_probe_openbmi_accpaper/run_*/`
- `j1/fold*_jepa.pt`
- `j0|j2|j2_random|j3/summary.json`（含 `delta_vs_shallow`）

决策：J2 相对 j2_random ≥ +0.03 再开 J3 / 跨窗门控。
