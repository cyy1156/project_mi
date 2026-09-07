# 5060 · teachable subset（方案 06）

旁路：可教试次清单 → 子集评估正式权重 → 可选子集 FT。  
**不**改写正式 5060 结果表。

## 入口

| 相 | 命令 |
|----|------|
| **B0** 清单 | `cd code/preprocess_lab` → `python -m src.datasets.openbmi.export_teachable_trials` |
| **B1** 评估 | `python eval_subset.py`（本目录；默认 shallow） |
| **B2** 微调 | `python finetune_subset.py`（需 B1 达门槛） |
| **P1 门控** | `python eval_gated.py`（伪在线 03 · OpenBMI 离线门控） |

产出清单：`find_best_trail/out/teachable_trials_v1.json` + `teachable_window_masks_v1.npz`  
评估/FT out：`code/train_lab/out/5060_teachable_subset_openbmi_accpaper/`  
记录 MD：`资料/模型训练/runs/5060_teachable_subset/`

本机 RTX 5060 请用仓库 `.venv`（`torch` cu128）：`D:\cyy\MI\.venv\Scripts\python.exe`。

## 示例

```bash
PY=D:/cyy/MI/.venv/Scripts/python.exe

# B0（全量约 ~30 min）
cd code/preprocess_lab
$PY -m src.datasets.openbmi.export_teachable_trials

# B1
cd ../train_lab/src/step/5060_teachable_subset_openbmi_accpaper
$PY eval_subset.py --model shallow --with-r3
$PY eval_subset.py --model eegnet --with-r3   # 可选

# B2（仅当 B1 Three R2−R0 达方案门槛）
$PY finetune_subset.py --task three --ft-mode head --max-folds 1

# P1 质量门控（产物写入 资料/伪在线实验/03_旁路_teachable质量门控/results/）
$PY eval_gated.py --model shallow --gates G0,G1,G2
$PY eval_gated.py --model shallow --gates G3   # 较慢：试次内 top-50%
$PY eval_gated.py --model eegnet --gates G0,G1,G2
```

正式权重默认：

- shallow `run_20260807_135828`
- eegnet `run_20260806_172218`
