# 03 · 2s/hop100 · Acc_paper 选模重训（仅 BCI2a T）

> **现行方案**：[`方案.md`](./方案.md) — **重训**；早停/选模 = **Val/Test Acc_paper**；train **batch balance**；**不用 E**。  
> **代码**：`code/train_lab/src/step/baselines_2s_hop100_accpaper/`  
> **历史复评（隔离）**：`baselines_2s_hop100_trialmaj/` + [`实验结果汇总_baselines_2s_hop100_trialmaj_bci2a.md`](./实验结果汇总_baselines_2s_hop100_trialmaj_bci2a.md)（**no_retrain**，非本版结论）

| 项 | 现行约定 |
|----|----------|
| 数据 | 仅 `A0*T` → `bci2a_2s_hop100` |
| 模型 | **全部 11** |
| 早停 | Val **Acc_paper** |
| 主报 | Test **Acc_paper** |
| 训练 | 窗 CE + **batch balance** |

```bash
cd code/train_lab/src/step/baselines_2s_hop100_accpaper
python run_all.py --continue-on-error
```

读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch no_E retrain=true`
