# 12 · LeJEPA Three 探测（BCI2a · 固定 4s）

| 项 | 路径 |
|----|------|
| **完整方案** | [`方案.md`](./方案.md) |
| 结果登记 | [`总结/结果登记表.md`](./总结/结果登记表.md) |
| **训练代码** | `code/train_lab/src/step/5060_lejepa_three_probe_bci2a_4s_accpaper/` |
| 数据 | `preprocess_lab/out/bci2a_4s`（`(N,1,8,1000)`） |
| 方法母本 | [`../11_旁路_LeJEPA_Three探测_openbmi_accpaper/`](../11_旁路_LeJEPA_Three探测_openbmi_accpaper/) |

相对 11：**仅**换 BCI2a + 固定 4s；LeJEPA / 方案 B / J0–J3 不变。下游早停 **Val BalAcc**。勿与 OpenBMI 0.540 绝对值混比。
