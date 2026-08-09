# 5060_shallow_mi_feat_openbmi_accpaper

Shallow · MI 特征工程旁路训练包（非正式表）。  
资料方案：`资料/模型训练/05_旁路_shallow_MI特征工程_openbmi_accpaper/`

| 脚本 | 臂 |
|------|-----|
| `baseline_shallow_a0.py` | A0 raw 8ch |
| `baseline_shallow_a1.py` | A1 + laterality 2ch |
| `export_trial_quality.py` | 导出 A2 权重 |
| `baseline_shallow_a2.py` | A2 A1 + 质量加权 |
| `baseline_shallow_a3.py` | A3 A1 + Mu 包络 |

```bash
cd code/train_lab/src/step/5060_shallow_mi_feat_openbmi_accpaper
python baseline_shallow_a0.py --max-folds 1
python baseline_shallow_a1.py
```
