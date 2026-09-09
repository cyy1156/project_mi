# experiment_game（在线系统 · 交稿精简树）

第一人称运动想象实验：视觉诱导 + OpenBCI 采集 / 打标 / 落盘 + 实时推理与采后微调。

**怎么跑**：见同目录 [`README_在线系统运行指南.md`](./README_在线系统运行指南.md)（或包根同名文件）。  
双击 [`open_operator.bat`](./open_operator.bat) → 浏览器打开操作台 Setup。

## 目录（与《00_目录结构与复现总览》一致）

| 位置 | 内容 |
|---|---|
| `tools/` | 启动与自检：`open_operator` / `preflight` / `open_induction` |
| `experiment/` | 会话状态机、在线推理、试次级读出、计分 |
| `config/` | 冻结配置：`v3_session.yaml`、`ft_policy.json`、`e1f_four_member.json`、`protocol.yaml` |
| `web/` | 操作台与游戏前端（纯 HTML/JS） |
| `acquisition/` · `core/` · `runtime/` | 采集门面与运行时支撑 |
| `offline/` · `pipeline/` | 采后 Phase4 / Leave-Next 微调 |
| `docs/` | 口径权威与协议摘录（`统计口径方案A_20260831.md` 等） |

交稿**不含**：内部实验脚本（`tools/exp*`）、机位笔记（`machines/`）、过程方案归档、真实被试 `data/`。

## 口径权威

- [`docs/统计口径方案A_20260831.md`](./docs/统计口径方案A_20260831.md) — 展示 / 门控 / 早停准确率口径  
- [`docs/ws_protocol.md`](./docs/ws_protocol.md) — WebSocket 协议  
- [`docs/marker_spec.md`](./docs/marker_spec.md) — Marker 表

**现行默认**：CausalFuse-8 四成员 · OpenBMI-Align 时序 · 采后 Leave-Next · 试次级因果平滑多数票读出。

依赖：包根 `requirements.txt`；权重路径见 `code/README_离线代码复现指南.md` §4。
