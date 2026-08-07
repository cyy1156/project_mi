# find_best_trail · OpenBMI MI 特征显著性实验

> 写入日期：2026-08-07  
> 目标：按重定标准，判断每个被试的 MI（运动想象）特征是否「明显」，为后续挑优试次 / 模板学习做基础。

## 1. 本目录文件

| 文件 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 实验目的、数据、重定标准、计分规则 |
| [`OpenBMI_25被试_MI特征显著性分析.md`](./OpenBMI_25被试_MI特征显著性分析.md) | **完整结果报告**（总表 + 分被试明细） |
| [`OpenBMI_25被试_MI特征显著性分析.json`](./OpenBMI_25被试_MI特征显著性分析.json) | 机器可读结果 |
| [`analyze_mi_features_25.py`](./analyze_mi_features_25.py) | 分析脚本副本（源在 `preprocess_lab/.../openbmi/`） |

参考文档（微信本地）：`脑电特征提取指标与模板量化分析.docx`  
同源备份亦在：`资料/模型训练/04_旁路_2s滑窗100ms_openbmi_accpaper/`

## 2. 一句话结论（subj01–25）

在 **OpenBMI_MI_feature_v1** 合格线下：

- **明显**：2 / 25（`subj03`、`subj18`）
- **中等**：13 / 25
- **弱/不明显**：10 / 25

多数被试达不到原文档「理想优秀模板」（Mu ERD −50%~−65%），但约六成达到本库「中等及以上」。

## 3. 如何复跑

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.analyze_mi_features_25
# 或指定被试
python -m src.datasets.openbmi.analyze_mi_features_25 --subjects 01,02,03
```

数据：`DATA/openbmi/openbmi/openbmi/sess*_subj*_EEG_MI.mat`  
范围：仅 `EEG_MI_train`；sess01+sess02 同人合并；当前已跑 **01–25**。
