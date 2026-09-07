# 原始验证数据包（交稿 · 仅指定集）

| 文件 | 说明 |
|---|---|
| `sample_submission.csv` | **正式提交预测**（官方模板已填写 `label∈{0,1,2}`，120 行，不得增删重排） |
| `submission_QuadFold59.csv` | 同内容内部备份（模型名归档；**邮件/评审请交 `sample_submission.csv`**） |
| `nested_N0_metrics.json` | 嵌套主要结果 Acc/召/特/F1 + 混淆矩阵 |
| `oof_N0/*.npy` | 嵌套验证折外预测（OOF）概率 / 标签 / 被试编号 |
| `官方数据说明.md` | 主办方《数据说明》副本 |
| `数据说明_使用对照.md` | 本队对照摘录 |
| `README.md` | 本说明 |

**加载提示**：`oof_N0_subjects.npy` 为 Unicode 字符串数组，`np.load(路径)` 默认参数即可读取（无需 `allow_pickle`）；`oof_N0_prob.npy` 形状 (900,3)、`oof_N0_y.npy` 形状 (900,)。
