# 10 · 5060 统一复现 + OTTA

> 状态：**代码已建 · 待跑** · v1.2  
> 详案：[方案.md](方案.md)  
> 代码：`code/train_lab/src/step/5060_stieger_unified_3s_hop100/`

## 快速开始

```powershell
# 1) 预处理（含 2s）
cd D:\MI\code\preprocess_lab
python -m src.datasets.stieger.batch_3s_hop100
python -m src.datasets.stieger.batch_2s_hop100

# 2) 冒烟
cd D:\MI\code\train_lab\src\step\5060_stieger_unified_3s_hop100
C:\Users\yy\.conda\envs\cyy\python.exe _smoke_local.py

# 3) Part II 2s 主线
.\run_part2_2s.ps1

# 4) Part III OTTA
python eval_otta.py --arms A0,A1,A2,A3,B0,B1,B2,B3,B4
python eval_c1.py
```

**注意**：2s 权重 `run_20260807_135828` 须存在于  
`train_lab/out/5060_baseline_openbmi_2s_hop100_accpaper/`。
