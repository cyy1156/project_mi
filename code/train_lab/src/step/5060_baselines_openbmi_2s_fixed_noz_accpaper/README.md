# 5060_baselines_openbmi_2s_fixed_noz_accpaper

方案 08：OpenBMI · **无 z-score** · 固定 Cue+[2,4)s · Acc_paper 十一模型。

- 数据：`preprocess_lab/out/openbmi_2s_fixed_cue2to4_noz/`
- 权重：`train_lab/out/5060_baseline_openbmi_2s_fixed_noz_accpaper/`
- 文档：`资料/模型训练/08_旁路_无zscore_固定窗_openbmi_accpaper/`

```bash
# 预处理
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_fixed_cue2to4 --limit 1
python -m src.datasets.openbmi.batch_2s_fixed_cue2to4

# 训练
cd ../train_lab/src/step/5060_baselines_openbmi_2s_fixed_noz_accpaper
python baseline_shallow.py --smoke
python run_all.py --continue-on-error
```
