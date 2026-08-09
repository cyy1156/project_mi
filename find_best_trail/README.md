# find_best_trail · OpenBMI MI 特征显著性实验

> 修订：2026-08-09  
> 目标：判断每个被试 MI 特征是否「明显」；服务后续挑优试次 / 模板学习。  
> 下游旁路：[`资料/模型训练/06_旁路_可教试次_子集评估_微调_openbmi_accpaper/`](../资料/模型训练/06_旁路_可教试次_子集评估_微调_openbmi_accpaper/)（B0 清单 → B1 子集评估 → 可选 B2 FT）。

## 1. 本目录文件

| 文件 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 切段几何、语料、54 人范围、计分规则 |
| [`analyze_mi_features_25.py`](./analyze_mi_features_25.py) | 转发入口 → `preprocess_lab` 源脚本 |
| `OpenBMI_54被试_MI特征显著性分析.md/.json` | v2 全量报告（跑完后生成） |
| `OpenBMI_25被试_MI特征显著性分析.md/.json` | 旧报告（历史；非正式） |

源脚本：`code/preprocess_lab/src/datasets/openbmi/analyze_mi_features_25.py`

## 2. 已确认约定

| 项 | 取值 |
|----|------|
| 语料 | OpenBMI **仅 `EEG_MI_train`**（不含 `EEG_MI_test`） |
| 被试 | **subj01–54**（默认全部） |
| Rest | Cue 前 **4 s**（可缩短避让上一 MI） |
| MI | Cue 后 **0–4 s** |
| 滑窗 | **2 s** / hop **100 ms**；**无**窗内 z-score |

## 3. 如何复跑

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.analyze_mi_features_25
# 子集
python -m src.datasets.openbmi.analyze_mi_features_25 --subjects 01,02,03

# 或在本目录
cd find_best_trail
python analyze_mi_features_25.py
```

## 4. B0 可教试次清单（teachable_v1）

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.export_teachable_trials
# 或：find_best_trail/export_teachable_trials.py
```

产出：`find_best_trail/out/teachable_trials_v1.json` · `teachable_window_masks_v1.npz` · README。  
下游评估：`code/train_lab/src/step/5060_teachable_subset_openbmi_accpaper/eval_subset.py`。

数据：`DATA/openbmi/openbmi/openbmi/sess*_subj*_EEG_MI.mat`  
输出默认：`find_best_trail/` + `资料/模型训练/04_5060_旁路_2s滑窗100ms_openbmi_accpaper/`
