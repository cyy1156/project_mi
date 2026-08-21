# 19 · Shallow 原版 vs 双频带分塔（T40 vs T20+T20）

> **V2 / V2_cat 已跑完**：V2 **0.5419±0.0223** ≈ V1(=S0) **0.5427±0.0243**（−0.08 pp）；V2_cat **0.5351±0.0239**（−0.76 pp）。详见 [总结/结果登记表.md](./总结/结果登记表.md)。

主对照（已确认）：

- **V1 = 方案 18 S0**（引用，不重跑）  
- **V2**：8–13 / 13–30 各 TemporalConv(**20**) → Spatial → **默认 Gate** 融合（本方案必跑）  
- **任务**：**仅 Three**（不训 Task 主表）  
- **超参**：与方案 18 `shared_hparams` 对齐（batch 256/512）  
- **包**：[`5070_dual_band_shallow_accpaper`](../../../code/train_lab/src/step/5070_dual_band_shallow_accpaper/)

详见 [方案.md](./方案.md)。
