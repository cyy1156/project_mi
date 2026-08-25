# 08 · 无 z-score · OpenBMI 固定 2s 窗（旁路）

OpenBMI · Cue **[2,4) s** Task + Cue 前 2 s Rest · **无滑窗** · **无 z-score** · Acc_paper 十一模型。  
目的：在「一试次一窗」下检验 ERD/ERS 可学性，并与 [07 滑窗无 z-score](../07_旁路_无zscore_2s滑窗_openbmi_accpaper/) 对照。

| 文档 | 说明 |
|------|------|
| [方案.md](方案.md) | 冻结协议（含新建数据管道） |
| [总结/结果登记表.md](./总结/结果登记表.md) | **已登记**（仅 shallow 完整；其余未齐） |

几何参考（BCI2a，非本臂数据）：[02_固定窗_bci2a_cue2to4s](../02_固定窗_bci2a_cue2to4s/)  
训练包：`code/train_lab/src/step/5060_baselines_openbmi_2s_fixed_noz_accpaper/`

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_fixed_cue2to4
cd ../train_lab/src/step/5060_baselines_openbmi_2s_fixed_noz_accpaper
python run_all.py --continue-on-error
```
