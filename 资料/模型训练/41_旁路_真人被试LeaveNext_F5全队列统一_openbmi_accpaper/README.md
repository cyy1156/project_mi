# 实验 41 · 真人被试 Leave-Next + F5 全队列统一

> 状态：**结构统一登记完成** · 2026-09-04  
> 目录：`资料/模型训练/41_旁路_真人被试LeaveNext_F5全队列统一_openbmi_accpaper/`  
> **主读**：[`总结/结果登记表.md`](总结/结果登记表.md)

## 一句话

把当前盘上 **全部 15 名真人被试** 的 session 结构、Leave-Next 爬坡键、以及 **最新 all4 + F5** 回放结果，收成同一套表。

## 入口

| 文档 | 用途 |
|------|------|
| [`方案.md`](方案.md) | 统一口径、纳入规则、与 31/33/34–40 关系 |
| [`总结/结果登记表.md`](总结/结果登记表.md) | **主读**：结构总表 + 末档排名 + 结论 |
| [`总结/详细结果_全队列分档与均值方差.md`](总结/详细结果_全队列分档与均值方差.md) | **每人每档明细** + 队列均值/样本标准差（本轮 `20260904_121803+`） |
| [`总结/detailed_stats.json`](总结/detailed_stats.json) | 机读统计 |
| [`总结/cohort_index.json`](总结/cohort_index.json) | 机读索引 |
| [`总结/_generated_cohort.md`](总结/_generated_cohort.md) | 分被试 session 全表 + 分档全表 |

## 再生

```text
conda activate cyy
python experiment_game/tools/_build_exp41_cohort.py
python experiment_game/tools/_build_exp41_registry.py
```

## 上游

- [`31`](../31_旁路_被试LeaveNext_F5读出_syj_fnz_openbmi_accpaper/) · [`33`](../33_旁路_真被试LeaveNext_all4复验_syj_fnz_openbmi_accpaper/)
- 工具：`experiment_game/tools/run_leave_next_e1f_task_ramp.py`
- 冻结：F5 读出 · 线上 FT=`all4+force`
