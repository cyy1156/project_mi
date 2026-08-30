# 实验 33 · 真被试 Leave-Next · so vs all4

完整方案：[`方案.md`](方案.md) · 结果：[`总结/结果登记表.md`](总结/结果登记表.md)

**状态：** **已完成**（2026-08-30）· stamp=`20260830_exp33`

- 被试：`syj0828` / `fnz0828`
- 臂：`e1f_so_force` / `e1f_all4_force`
- 主指标：F5（因果平滑 lookback=2 + 多数票）试次 MI acc

## 末档速览

| 被试 | so MI | all4 MI | Δ |
|------|-------|---------|---|
| syj0828 | 0.222 | **0.944** | **+0.722** |
| fnz0828 | 0.056 | **0.417** | **+0.361** |

两人均 all4≫so · 平均 ΔMI ≈ **+0.54** → **支持线上默认 all4 + force**（堵住「仅仿真 Exp32」）。

```text
python experiment_game/tools/run_real_subject_all4_vs_so.py --all --stamp 20260830_exp33
```

原始：`experiment_game/data/subjects/_analysis/exp33_real_all4_20260830_exp33/`
