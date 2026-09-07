# 原始验证数据包

| 文件 | 说明 |
|---|---|
| `sample_submission.csv` | **正式提交预测**（官方模板已填写 `label∈{0,1,2}`，120 行） |
| `submission_QuadFold59.csv` | 同内容内部备份（模型名归档） |
| `nested_N0_metrics.json` | QuadFold-59 留一被试主要结果 Acc/召/特/F1 + 混淆矩阵 |
| `数据说明_使用对照.md` | 与官方数据说明对照 |
| `oof_N0/*.npy` | `oof_N0_subjects.npy` 为 Unicode 字符串数组，`np.load()` 默认参数可读；prob 形状 (900,3)、y 形状 (900,) |

复现留一被试指标：加载 `oof_N0_*.npy` 折外预测或直接读本目录 JSON。  
**邮件/评审请提交已填写的 `sample_submission.csv`，不要另交单独命名的预测文件。**
