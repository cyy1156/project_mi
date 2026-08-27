# 27 · 旁路 · fnz 被试 FT + Replay 方案对比（OpenBMI 3s）

> 旁路深研 · **非正式夺冠表** · 2026-08-27 · **待跑**  
> 目的：在 fnz 真实采集数据上，对比 **Replay 池 / replay_ratio / 冻 head**，为 `experiment_game` 被试专用 FT SOP 定默认值。  
> 关联采集方案：[`../../../experiment_game/docs/fnz实验与微调采集方案_20260827.md`](../../../experiment_game/docs/fnz实验与微调采集方案_20260827.md)

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 双轨 grid · 14 arm × 2 track · 验收与决策 |
| [`总结/结果登记表.md`](./总结/结果登记表.md) | 跑完后填数 |
| Replay 质量依据 | [`../../../find_best_trail/`](../../../find_best_trail/) |
| 可教清单 | `find_best_trail/out/teachable_trials_v1.json` |
| 前序 teachable 旁路 | [`../06_旁路_可教试次_子集评估_微调_openbmi_accpaper/`](../06_旁路_可教试次_子集评估_微调_openbmi_accpaper/) |

### 双轨（必做）

| Track | 训练 session | 场景 | 训练窗 / heldout |
|-------|--------------|------|------------------|
| **S · single** | **ws01 单独** | 仅 1 次合格 session 后 FT（≈ FT1） | **208 / 102** |
| **M · merge** | **ws01 + ws02** | 两次 session 合并 FT（≈ FT2 / 当前上线） | **338 / 152** |

每条 track 各跑 **14 个 arm** → 共 **28 次** FT。

### 怎么跑（拍板后）

```bash
cd d:\MI
# 若无 teachable 清单
cd code/preprocess_lab && python -m src.datasets.openbmi.export_teachable_trials && cd ../..

python experiment_game/tools/exp27_fnz_replay_grid.py
# 可选：--tracks S,M --arms A0,B2,C1
```

产出：`experiment_game/data/models/fnz/exp27/` + 本目录 `总结/结果登记表.md`

### 与正式表关系

| 项 | 说明 |
|----|------|
| 底座 | 正式 5070 OpenBMI 3s shallow（`run_20260822_094942`） |
| 目标 | fnz **heldout 个性化** + **无类塌缩** |
| 不改 | `资料/实验结果/5060/` 十一模型表 |
