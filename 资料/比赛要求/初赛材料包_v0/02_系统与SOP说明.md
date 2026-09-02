# 02 · 系统与 SOP 说明（初赛 v0）

> **权威优先级**：配置 + 代码 > `experiment_game/docs/框架冻结确认_20260829.md` > 其它文档。  
> 本文件为评委可读摘要，不替代冻结原文。

---

## 1. 系统组成

| 模块 | 作用 | 仓库位置（相对根） |
|------|------|-------------------|
| 采集 / LSL | 实时脑电接入、预处理入口 | `collect_data/`、`experiment_game/acquisition/` |
| 会话与游戏 | 探针/采集/交互、事件落盘 | `experiment_game/` |
| 在线推理 | E1f 融合 + F5 读出/计分 | `experiment_game` 推理与 scoring 配置 |
| 离线训练 | OpenBMI 预训练与旁路实验 | `code/train_lab/`、`code/preprocess_lab/` |
| 被试资产 | 身份、权重 current、FT runs | `experiment_game/data/subjects/{id}/` |

---

## 2. 现行 SOP 冻结要点

| ID | 结论 |
|----|------|
| F1 | 正式时序 = **OpenBMI-Align v1** |
| F2 | Rest = 专用试次间 Rest **4s**（不含其后 prep 2s） |
| F3 | 零样本默认底座 = **E1f 四成员** |
| F4 | 四成员 **含 T-shallow** |
| F5 | **单轨读出**：因果平滑 lookback=2 + 多数票；τ 早停退出正式 SOP |
| F6 | MI 对 +1；Rest 对 +0.5 |
| F7 | 微调主路径 = **采后 Leave-Next** |
| F8 | 门控 FAIL 可警告后强制晋升 |
| F9 | OpenBMI replay 默认 **0.10** |
| F19 | 比赛技术方案与现场 SOP **允许分叉** |
| F22 | 训练旁路数字作证据；**部署以配置为准** |
| F25 | 通道序：`FC3, C3, CP3, CZ, CPZ, FC4, C4, CP4` |
| F26 | 切窗：`openbmi_hop100` · 3s/hop100 · Cue 前 0.5s 基线 |

### 时序口述

```text
[专用 Rest 4s] → [prep 2s] → [Cue = MI onset] → [MI 4s] → [ITI 3s] → …
```

### 读出口述（F5）

```text
窗：E1f 融合 → 因果平滑(n,n−1,n−2) → argmax
试次：多数票 = 主判定 = 计分依据
```

---

## 3. 与比赛需求能力对照

| 比赛需求 | 声明 | 证据文件 |
|----------|------|----------|
| 左/右/静息实时分类 | ✅ 已具备 | 本文件 + 演示视频（待录） |
| 完整采集—解码—交互 | ✅ 已具备 | experiment_game 冻结链路 |
| 双向适配（引导+算法） | ✅ 部分～具备 | 游戏 + Leave-Next（`04`） |
| ≥20 人自采 | 🔄 进行中 | `05` |
| 可验证可复现 | ✅ 有公开集+真人数字 | `03` `04` `08` |

---

## 4. 操作与复现入口（对内）

| 用途 | 文档 |
|------|------|
| 冻结总表 | `experiment_game/docs/框架冻结确认_20260829.md` |
| 范式对齐 | `experiment_game/docs/范式对齐_OpenBMI与fnz_v3_20260827.md` |
| 配置权威 | `v3_session.yaml` / `e1f_four_member.json` / trial_scoring 相关配置 |

---

*初赛材料包 v0 · 2026-08-30*
