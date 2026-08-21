# 20 · Shallow · OpenBMI 3s/hop100 Acc_paper

> 任务方案见 [`方案.md`](./方案.md)  
> 状态：**S3 五折已完成** · Task **0.7415±0.0306** · Three **0.5876±0.0296**（vs 2s +4.7 pp）

## 一句话

其他条件对齐正式 `5060_baselines_openbmi_2s_hop100_accpaper` 的 **Shallow**，只把窗长改为 **3 s / hop 100 ms**；正式五折显示相对 2s **Task/Three 均约 +4.7 pp**。

## 对照与本臂

| 臂 | Tw | Task Acc_paper | Three Acc_paper |
|----|-----|----------------|-----------------|
| S2（正式） | 2s | 0.6941±0.0349 | 0.5404±0.0256 |
| **S3** | **3s** | **0.7415±0.0306** | **0.5876±0.0296** |

登记详情：[`总结/结果登记表.md`](./总结/结果登记表.md)

## 本臂入口

| 项 | 路径 |
|----|------|
| 预处理 | `python -m src.datasets.openbmi.batch_3s_hop100` |
| 数据 | `preprocess_lab/out/openbmi_3s_hop100/` · `(178200,1,8,750)` |
| 训练 | `5060_baselines_openbmi_3s_hop100_accpaper/baseline_shallow.py` |
| 正式 run | `out/.../run_20260821_190504/` |
