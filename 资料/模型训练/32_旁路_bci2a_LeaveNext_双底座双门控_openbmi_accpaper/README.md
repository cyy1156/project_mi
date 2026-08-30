# 实验 32 · BCI2a Leave-Next · 双底座 × 双门控 × FT 范围

完整方案：[`方案.md`](方案.md) · 结果：[`总结/结果登记表.md`](总结/结果登记表.md)

**状态：** **已完成**（2026-08-30）· stamp=`20260829_235900` · **A01–A09 齐全**

- **P0**：Shallow 单头 vs **E1f（只 FT shallow + 冻结三员融合）** × strict/force  
- **P1**：E1f 内 **so** vs **四员各自 FT 再融合（all4）** × 门控  

主指标 = **三分类窗级 acc（含 Rest）** + 因果平滑 lookback=2。

## R5 九人均值（主结论速览）

| 臂 | R5 mean |
|----|---------|
| shallow × force | **0.678** |
| e1f_so × force | 0.445 |
| e1f_all4 × force | **0.671** |

- P1：all4 − so = **+0.225**（9/9 同向）→ **过 +2pp 门槛，倾向四员各自 FT**  
- P0：现行 so 融合读出在 Leave-Next 上弱于单头 shallow；all4 可回到与 shallow 同档  

**线上已改（2026-08-30）**：`experiment_game/config/ft_policy.json` 默认 **`ft_scope=all4`** + **`force_promote_on_gate_fail`/`auto_promote_after_ft`**；产物含 `members/` + `e1f_overlay.json`，FAIL 时告警落盘仍晋升。
## 怎么复跑

```text
python experiment_game/tools/run_exp32_leave_next_dual_model_gate.py --all --phase all --stamp 20260829_235900
```

原始：`experiment_game/data/sim_subjects/_analysis/exp32_20260829_235900/`

相关：[`实验 29`](../29_旁路_bci2a被试FT_replay验证_openbmi_accpaper/) · [`实验 31`](../31_旁路_被试LeaveNext_F5读出_syj_fnz_openbmi_accpaper/) · 冻结 F5/F7/F8
