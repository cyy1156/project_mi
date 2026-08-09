# 03 · teachable 质量门控（旁路）

> 旁路 · **非正式夺冠表** · 2026-08-09  
> **P1 阳性 / P2 现行与 04 对齐（OpenBMI · 阳性）**；历史 BCI2a-P2 阴性已冻结对照。

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | P1 OpenBMI 离线 → P2 游戏在线（默认 OpenBMI） |
| [`总结/结果登记表.md`](./总结/结果登记表.md) | 主报登记 |

### 两阶段

| 相 | 内容 | 质量分来源 |
|----|------|------------|
| **P1** | OpenBMI hop100 复评 + mask 门控 | `teachable_v1` 窗 mask |
| **P2** | 游戏会话在线门控（**OpenBMI shallow + 通道重排**，同 04） | 同 trial REST → ERD/laterality |

**禁止**覆盖 01/02 的 `results/`。

### 与 01/04 关系

| 臂 | 本臂 |
|----|------|
| 01 BCI2a 零样本 | 历史 P2 对照（`--weight-domain bci2a`） |
| **04 OpenBMI 零样本+门控** | **P2 现行同协议**；部署口径一致 |
| 06 B1/B2 | 动机与阈值来源；B2 头默认不进 P2 |

### P1 怎么跑

```bash
PY=D:/cyy/MI/.venv/Scripts/python.exe
cd code/train_lab/src/step/5060_teachable_subset_openbmi_accpaper
$PY eval_gated.py --model shallow --gates G0,G1,G2
```

### P2 怎么跑（默认 OpenBMI · 与 04 对齐）

```bash
PY=D:/cyy/MI/.venv/Scripts/python.exe
cd code/train_lab/src/step/game_pseudo_online_hop100
$PY eval_gated_p2.py --model shallow --gates H0,H1,H2,H3
# 历史 BCI2a 对照：
$PY eval_gated_p2.py --model shallow --weight-domain bci2a --gates H0,H1,H2,H3
```

产物：`results/<stamp>_*_P2_openbmi_gate.md`（或 `_P2_bci2a_gate.md`）

上级索引：[../README.md](../README.md)
