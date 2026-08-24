# v2 实施看板

> 随开发更新 · 规格以 [`v2_upgrade_plan.md`](v2_upgrade_plan.md) 为准

| 模块 | 状态 | 备注 |
|------|------|------|
| `trial_v2.py` | ✅ | D8 栅格/计分/早停/无效；`score_reach` 取代 reach |
| `session_v2.py` | ✅ | D8 连 5 无效熔断 → 可选 FT → v1 降级 |
| `SessionRunner.run_v2_session` | ✅ | |
| `orchestrator._run_v2_session` | ✅ | auto phase4 双管道 |
| `inference_v2.py` | ✅ M2 | mi_start 锚点 |
| `v2_session.yaml` / `.pilot.yaml` | ✅ | 操作台选 preset 待做 |
| `operator.html` v2 | 🟡 | 模式+引导+gate 文本；无曲线图/模拟 touch |
| `v2_bridge.js` demo | ✅ | D8 `score_reach` 演示 |
| `scene.js` __v2scene | 🟡 | 基础挂点；无碰撞 |
| phase4 merge / pooled | ❌ | D5=A、D3=C 待实现 |
| 标定 FT 异步 + ft_lagged | ❌ | D6=B |
| `quiz_curve.csv` 导出 | ❌ | |
| touch / 模拟 touch WS | ❌ | |
| **D8 在线判定 v2.1** | ✅ | §3.4 + `scoring_replay.py` 离线回放 |
| `v2_session.yaml` D8 字段 | ✅ | `V2Config` + trial/session 已读 |
| 标定 FT 异步 | ❌ | 当前同步 |
| DriftGuard 接线 | ⏸ | |
| `v2_session_mode_spec` 同步 | ✅ 2026-08-24 | v2 标准重写 |
| `采集流程_20被试底座v2` 同步 | ✅ 2026-08-24 | D8 / touch / 三管道 |
| 文档对齐（开发计划/系统设计/提交方案） | ✅ 2026-08-24 | 见定稿 §11；提交方案改写前已备份 |

**冒烟：** L0 未正式跑 · L1 未签字 · L2 未做
