# 5060_baselines_openbmi_3s_hop100_accpaper

OpenBMI · **3s/hop100** · Acc_paper · **仅 Shallow**（实验 20）。  
相对正式 2s 基线包：只改窗长；其余协议/HP 冻结。

## 正式五折（RTX 5060 · Fast · `run_20260821_190504`）

| 头 | Test Acc_paper | vs 正式 2s |
|----|----------------|------------|
| Task | **0.7415±0.0306** | **+4.74 pp** |
| Three | **0.5876±0.0296** | **+4.72 pp** |

对照 2s：Task 0.6941±0.0349 · Three 0.5404±0.0256

| 项 | 路径 |
|----|------|
| 任务方案 | `资料/模型训练/20_旁路_shallow_3s滑窗100ms_openbmi_accpaper/方案.md` |
| 数据 | `preprocess_lab/out/openbmi_3s_hop100/` · `(178200,1,8,750)` |
| 权重 | `out/5060_baseline_openbmi_3s_hop100_accpaper/.../run_20260821_190504/` |

## 训练

```powershell
cd code/train_lab/src/step/5060_baselines_openbmi_3s_hop100_accpaper
D:\cyy\MI\.venv\Scripts\python.exe baseline_shallow.py --max-folds 0 --num-workers 0
```

## 禁止

- 覆盖 `out/5060_baseline_openbmi_2s_hop100_accpaper/`
- 把 3s 写入正式十一模型表
