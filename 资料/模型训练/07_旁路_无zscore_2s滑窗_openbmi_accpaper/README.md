# 07 · 无 z-score · OpenBMI 2s/hop100 滑窗（旁路）

正式 Acc_paper 滑窗同形，**唯一差异：窗内 z-score 关闭**。  
目的：看十一模型是否更利用 **ERD/ERS 功率** 信息。

| 文档 | 说明 |
|------|------|
| [方案.md](方案.md) | 冻结协议 + ERS 探针 |
| [总结/结果登记表.md](总结/结果登记表.md) | 待填 |

姊妹臂：[08 固定窗无 z-score](../08_旁路_无zscore_固定窗_openbmi_accpaper/)  
正式对照：[04_5060 OpenBMI Acc_paper](../04_5060_旁路_2s滑窗100ms_openbmi_accpaper/)  
训练包：`code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_noz_accpaper/`

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --no-zscore
cd ../train_lab/src/step/5060_baselines_openbmi_2s_hop100_noz_accpaper
python run_all.py --continue-on-error
```
