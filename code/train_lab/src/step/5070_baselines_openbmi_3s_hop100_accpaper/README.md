# 5070_baselines_openbmi_3s_hop100_accpaper

OpenBMI · **3s/hop100** · Acc_paper · **仅 Shallow**（实验 **21** · RTX 5070）。  
协议同实验 20（5060）；本包独立 `out/5070_*`，**禁止**写入 `5060_*`。

## 相对 5060 实验 20

| 项 | 5060（实验20） | **5070（本包）** |
|----|----------------|------------------|
| 机位 | RTX 5060 Laptop | **RTX 5070 Laptop** |
| out | `5060_baseline_openbmi_3s_hop100_accpaper` | **`5070_baseline_openbmi_3s_hop100_accpaper`** |
| batch | 128/256 | **128/256**（对齐 5070 计划 §3） |
| 数据 / 模型 / Acc_paper | 同 | **同** |

## 训练

```powershell
cd code/train_lab/src/step/5070_baselines_openbmi_3s_hop100_accpaper
# 建议先 fold0
D:\cyy\MI\.venv\Scripts\python.exe baseline_shallow.py --max-folds 1 --num-workers 0
# 正式五折
D:\cyy\MI\.venv\Scripts\python.exe baseline_shallow.py --max-folds 0 --num-workers 2
```

| 项 | 路径 |
|----|------|
| 任务方案 | `资料/模型训练/21_5070_旁路_shallow_3s滑窗100ms_openbmi_accpaper/` |
| 数据 | `preprocess_lab/out/openbmi_3s_hop100/` |
| 权重 | `out/5070_baseline_openbmi_3s_hop100_accpaper/.../` |
| 5060 对照 | `run_20260821_190504`（只读） |

## 禁止

- 覆盖 / 续写 `out/5060_*`
- 把 3s 写入正式十一模型 2s 表
