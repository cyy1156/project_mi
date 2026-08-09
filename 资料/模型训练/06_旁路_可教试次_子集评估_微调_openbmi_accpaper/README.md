# 06 · 可教试次 · 子集评估 · 质量微调（OpenBMI Acc_paper）

> 旁路深研 · **非正式夺冠表** · 2026-08-09 · **B0/B1/B2 已跑完**  
> 承接 05（通道 FE 阴性）：改做 **清单 / 条件评估 / 可选子集 FT**。  
> 数字见 [`总结/结果登记表.md`](./总结/结果登记表.md)。  
> 正式十一模型仍以 [`../04_5060_旁路_2s滑窗100ms_openbmi_accpaper/`](../04_5060_旁路_2s滑窗100ms_openbmi_accpaper/) 为准。

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 协议冻结 · B0→B1→B2 · 决策门槛 |
| [`总结/结果登记表.md`](./总结/结果登记表.md) | B1/B2 数字登记 |
| 特征依据 | [`../../../find_best_trail/`](../../../find_best_trail/) |
| 前序阴性 | [`../05_旁路_shallow_MI特征工程_openbmi_accpaper/`](../05_旁路_shallow_MI特征工程_openbmi_accpaper/) |

### 三相

| 相 | 内容 | 是否改权重 |
|----|------|------------|
| **B0** | 可教试次清单（真 ERD/laterality） | 否 |
| **B1** | 明显 12 + 高 lat 试次上评正式权重 | 否 |
| **B2** | 仅当 B1 达门槛：冻骨干/小 lr 子集 FT | 是（旁路 out） |

### 与 05 差异

| 项 | 05 | 06 |
|----|----|----|
| 杠杆 | 输入通道 / 全库样本权 | 选样 · 条件评估 · 子集 FT |
| 质量分 | 窗内假代理（A2） | 分析臂真 laterality/ERD |
| 正式表 | 不改 | 不改 |

### 怎么跑

```bash
# B0 可教清单
cd code/preprocess_lab
python -m src.datasets.openbmi.export_teachable_trials
# 产出：find_best_trail/out/teachable_trials_v1.json + teachable_window_masks_v1.npz

# B1 子集评估正式权重
cd ../train_lab/src/step/5060_teachable_subset_openbmi_accpaper
python eval_subset.py --model shallow
python eval_subset.py --model eegnet --with-r3   # 可选

# B2（仅当 B1 Three R2−R0 达方案门槛）
python finetune_subset.py --task three --ft-mode head
```

代码：`code/train_lab/src/step/5060_teachable_subset_openbmi_accpaper/`  
记录：`资料/模型训练/runs/5060_teachable_subset/`
