# 5060_baselines_openbmi_2s_hop100_noz_accpaper

方案 07：OpenBMI · **无 z-score** · 2s/hop100 · Acc_paper 十一模型。

- 数据：`preprocess_lab/out/openbmi_2s_hop100_noz/`
- 权重：`train_lab/out/5060_baseline_openbmi_2s_hop100_noz_accpaper/`
- 对照代码：`5060_baselines_openbmi_2s_hop100_accpaper/`（有 z-score 正式表）
- 文档：`资料/模型训练/07_旁路_无zscore_2s滑窗_openbmi_accpaper/`

```bash
# 预处理
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --no-zscore --limit 1   # 冒烟
python -m src.datasets.openbmi.batch_2s_hop100 --no-zscore

# 训练
cd ../train_lab/src/step/5060_baselines_openbmi_2s_hop100_noz_accpaper
python baseline_shallow.py --smoke
python run_all.py --continue-on-error
```
