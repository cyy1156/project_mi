# experiment_game · v2 完整升级计划（整合版）

> 版本：**v1.5** · 2026-08-24  
> **规格源：** 本文 §1–§3 + [`config/v2_session.yaml`](../config/v2_session.yaml)  
> **实施看板：** [`v2_backlog.md`](v2_backlog.md)（模块状态，随开发更新）  
> **试点 preset：** [`config/v2_session.pilot.yaml`](../config/v2_session.pilot.yaml)

---

## 0. 升级目标（一句话）

在 **不推翻** v1 基建的前提下，把「6s 想象 + 标定轮增量适配 + 准入 + 第一人称游戏反馈」嵌进 **一次被试会话**；入口 `SessionRunner.run_v2_session()`。

---

## 1. 冻结决策（2026-08-23 / **D8 · 2026-08-24** 用户确认 · 不可单方改）

| # | 议题 | 决策 |
|---|------|------|
| 1 | Cue 与 MI | Cue **2s** → 结束后 `mi_start`；在线判定 **v2.1**（§3.4）：`mi_start` 后 **0.6s 栅格**至 6.0s，**加权分**计分（取代旧 3/4/5/6s 四定点） |
| 2 | 有反馈阶段数据 | 游戏 **不必满 6s**；`mi_end` = touch/超时；离线 **`phase4_v2_game`** |
| 3 | 游戏视觉 | **第一人称**；左/右 Cue → **同侧手**抓桌物；**碰物或 6s** 结束 |
| 4 | 游戏训练 | **试次内推理冻结**；**每轮游戏结束后** `train_round` |
| 5 | 漂移保护 | **暂缓** |
| 6 | 入口 | `SessionRunner.run_v2_session()` |
| 7 | 参数 | **统一 yaml**；`game_rounds`、**D8 计分参数** 等可配，不写死 |

### 1.1 实施决策（已确认）

| # | 决策 |
|---|------|
| **D1** | **A** — touch 后立即 `mi_end`，不再跑后续判定点 |
| **D2** | **B** — 游戏轮间 FT：**全部**游戏试次窗入训（不设最短 mi 门槛） |
| **D3** | **C** — 游戏短片段走 **fragment 拼接池**（见 §3.5）；标定仍按满段切窗 |
| **D4** | **A** — 先做操作台 **模拟 touch**，再接 scene 碰撞 |
| **D5** | **A** — 会话结束自动跑三管道 + 写 `phase4_v2_merged/`（pooled 旁路记入 manifest，默认不并入主训练池） |
| **D6** | **B** — 标定轮间 **异步** FT；下轮用上轮结束 ckpt；未完成则 `ft_lagged` 事件 |
| **D7** | **A** — `weak_mi` 与 pass 相同 `game_rounds`，全程标记 `weak_mi` |
| **D8** | **v2.1 在线判定** — 0.6s 栅格 + 半票/满票加权分；Score≥5 早停；错类≥5 或 Score≤3 无效；连 5 无效熔断（§3.4） |

---

## 2. 单试次时间线（墙钟）

```text
0–2s    prep
2–4s    Cue 动画（MI 未开始）
4s      mi_start
4–10s   MI ≤6s（判定点：mi+0.6, 1.2, …, 6.0s，步长 0.6s）
        标定：无早停/无错类作废时 **恒满 6s** → mi_end
        游戏：touch（D1）或 Score≥5（D8）或错类作废（D8）或满 6s → mi_end
+3s     ITI
```

最长 **13s/trial**（prep 2 + cue 2 + mi 6 + iti 3）。**最早成功早停**：mi_start **+4.2s**（0.6…4.2s 连续判对，加权分达 5.0）。

---

## 3. 端到端数据契约（核心 · v1.2 新增）

> 标定与游戏在「展示 / 脑电 / 在线窗 / FT / 切窗」五列必须一致；实现与验收均对照本表。

| 维度 | 标定轮 | 游戏轮 |
|------|--------|--------|
| **展示结束** | 无早停时 mi 满 6s → ITI | touch / **Score≥5**（D8）/ 错类作废 / 满 6s → ITI |
| **脑电段** | `[mi_start, mi_start+6s]` | `[mi_start, mi_end]`，`mi_end`≤mi+6s |
| **在线判定** | **v2.1** §3.4：mi 后每 **0.6s** 一档至 6.0s；可 **离线回放**计分 | 同左；**touch 后立即 `mi_end`**（D1），不再后续档位 |
| **入标定 FT** | 每轮前 12 试次窗 | — |
| **入小考** | 每轮后 6 试次，**永不进 FT** | — |
| **入游戏 FT** | — | 轮间：**全部**试次窗入训（D2=B） |
| **离线切窗** | `phase4_v2` | `phase4_v2_game` + **`phase4_v2_game_pooled`**（D3=C） |
| **短试次** | mi 恒 ≥6s | 单试次 &lt;3.4s → 进 **fragment 池** 拼接；仍切不出则丢弃 |

### 3.1 游戏结束：过渡 vs 目标

| 阶段 | 结束条件 | 前端 | 后端事件 | 状态 |
|------|----------|------|----------|------|
| **过渡（已废弃）** | 同 label 连判对 **4 次** | `arm_level` 升档 | `reach` | 已由 D8 取代 |
| **D8（已实现）** | 真标签 **加权分 ≥5.0** 或 **touch** | 得分/伸手反馈 | `score_reach` 或 `touch` → `mi_end` | ✅ `trial_v2` |
| **目标** | **碰撞 touch** 桌物 | 同侧手 Raycast/AABB | `touch` → `mi_end` | 待做（D1） |
| **调试捷径** | 操作台 **模拟 touch** 键 | — | 注入 `touch` | **D4=A，先做** |

`arm_levels` / `v2_ui_refit_plan_A` 四档升物 **作废**；事件名 `arm_level` 可保留至 D8 接线完成。

### 3.4 在线判定 v2.1（D8 · 2026-08-24 冻结）

> **不要求**试次内实时 forward；允许用录制 EEG + `mi_start` 时间戳 **离线回放**同一规则计分。事件须落盘每档 `t_k`、`pred`、`score`、`invalid_reason`。

#### 判定点栅格

相对 **`mi_start`**（Cue 2s 结束后立即开始）：

```text
t_k = 0.6 · k  s，k = 1 … 10  →  0.6, 1.2, …, 6.0
```

每档 **1 票**（3s 窗、hop100 与 S3 训练同构；`t_k ≤ 2.4` 时 3s 窗可跨 Cue，仅作弱证据）。

#### 票权（对/错对称）

| 档位 | t_k (s) | 权重 w_k |
|------|---------|----------|
| 半票段 | 0.6, 1.2, 1.8, 2.4 | **0.5** |
| 满票段 | 3.0, 3.6, 4.2, 4.8, 5.4, 6.0 | **1.0** |

- **真标签分 Score**：pred = label → **+w_k**；pred ≠ label → 真标签 **+0**
- **错类累计**：pred ≠ label → 该 pred 类 **+w_k**（与真标签分对称加权）

#### 试次内判定顺序

1. 按栅格逐档计分（或事后回放）。
2. **早停成功**：任意时刻 **Score ≥ 5.0** → `mi_end`（`reason=score_5`）。最早 **mi_start + 4.2s**（0.6…4.2s 全对）。
3. **无效 A**：任一错误类加权累计 **≥ 5.0** → `trial_invalid_wrong_race`，`mi_end`。
4. 若未触发 2/3：**必须跑满** `mi_start + 6.0s` → `mi_end`。
5. **无效 B**：最终 **Score ≤ 3.0** → `trial_invalid_low_score`；**Score ≥ 4.0** → 有效（未满 5 亦可）。

理论满分（10 档全对且打满 6s）：**8.0**（4×0.5 + 6×1.0）。

#### 会话熔断与降级

| 条件 | 行为 |
|------|------|
| **连续 5 个 trial** 为无效 A 或 B | 停止 v2 会话；`v2_abort_reason=consecutive_invalid_5` |
| 熔断时本场 **有效 trial 窗数 ≥ K**（默认 **K=6**） | 用有效窗做 **一轮增量 FT** |
| 有效窗 **< K** | **仅降级，不 FT** |
| 降级 | 退回 **v1 诱导/记录**（无在线 `judgment_fn`）；events 落盘；操作员选择是否重开 v2 |

**不计入**连 5 无效：操作员 `trial_reject`、LSL 无数据、引导失败。

#### 配置（当前无操作台 UI）

| 参数 | 默认 | 说明 |
|------|------|------|
| `judgment_step_s` | 0.6 | 栅格步长 |
| `judgment_half_weight_until_s` | 2.4 | 半票截止（含） |
| `score_early_stop` | 5.0 | 早停阈值 |
| `score_invalid_max` | 3.0 | ≤ 无效 |
| `score_valid_min` | 4.0 | ≥ 有效 |
| `wrong_class_abort` | 5.0 | 错类累计作废 |
| `consecutive_invalid_abort` | 5 | 会话熔断 |
| `ft_min_valid_trials` | 6 | 熔断后 FT 门槛 |

来源：[`v2_session.yaml`](../config/v2_session.yaml)（或 `run_config.experiment.v2_config_path` 指向的 preset）；**改 YAML 后重开会话**生效。实现：`adapt_engine/scoring_v21.py`、`trial_v2.py`、`session_v2.py`；离线回放见 `scoring_replay.py`。

#### 与 D1 touch 的关系

- **touch** 优先：一旦发生，`mi_end` 立即结束，**不再**跑后续档位；该 trial Score 按已计档位结算。
- 游戏展示可与 Score 联动；**reach 临时逻辑（连判对 4 次）在 D8 接线后废弃**。

### 3.2 配置 preset

| 文件 | 用途 |
|------|------|
| [`v2_session.yaml`](../config/v2_session.yaml) | **正式采集**（默认） |
| [`v2_session.pilot.yaml`](../config/v2_session.pilot.yaml) | **试点/冒烟**（少轮、短 gap、少 ft_epochs） |

操作台后续支持选择 preset；当前可 `run_v2_session(config_path=".../v2_session.pilot.yaml")`。

**`cal_rounds_min` 语义：** 设计默认轮数，**不**强制最少标定轮才准入（第 2 轮小考满 6 且 ≥60% 可 pass）。

### 3.3 会话结束科研产物（自动落盘）

| 文件 | 内容 |
|------|------|
| `session.meta.json` | `phase_mode=v2_session`, `v2_summary` |
| `events.jsonl` | 含 `mi_start/mi_end/touch/reach/judge`（每档 `t_k/score`）、`trial_invalid_*`、`v2_abort_*`、`v2_guidance_*` |
| `alignment/trial_table.csv` | `phase`, `t_mi_*`, `mi_dur` |
| `v2_ckpts/` | 轮间 ckpt |
| **`quiz_curve.csv`**（待实现） | `round,k_ft,n_quiz,acc` → 对接实验 25 登记表 |
| `phase4_v2/` + `phase4_v2_game/` + `phase4_v2_game_pooled/` | 标定 / 游戏单试次 / fragment 池 → **D5=A** 自动 `phase4_v2_merged/` |

### 3.5 游戏短片段 · fragment 拼接池（D3=C · 已确认）

**仅游戏阶段、仅离线**；与标定 `phase4_v2` 及 openbmi 默认同构训练**分目录**。

| 规则 | 说明 |
|------|------|
| 输入 | `phase=game` 且 `mi_dur < 3.4s`（单试次切不出 3s 窗）的 `[mi_start, mi_end]` |
| 拼接 | **同 label**（左+左 / 右+右）、**同一 game_round** 内，按 `trial_id` 顺序首尾相接 |
| 最短片段 | 单段 `mi_dur ≥ 1.0s` 才进池；更短丢弃 |
| 禁止 | 跨 ITI、跨轮、跨 label、混入静息 |
| 输出 | `phase4_v2_game_pooled/`；manifest `mode=fragment_pool` |
| 滑窗 | 在拼成的连续序列上再做 3s hop100（与 openbmi 同滤波/z-score） |
| 单试次够长 | 仍走 `phase4_v2_game`（不重复进 pooled） |

### 3.7 weak_mi 游戏（D7=A · 已确认）

- 6 轮标定仍未达 `gate_enter_three` → 状态 `weak_mi`，权重取历史最优 ckpt  
- **游戏轮数** = 与 pass 被试相同 `game_rounds`（不减少、不跳过）  
- `session.meta.json` / events 标 `weak_mi`；登记表单独一列  

### 3.8 phase4 合并（D5=A · 已确认）

会话结束（`orchestrator` + `auto_phase4`）顺序：

1. `phase4_v2.py` → `phase4_v2/`  
2. `phase4_v2_game.py` → `phase4_v2_game/`  
3. `phase4_v2_game_pooled.py` → `phase4_v2_game_pooled/`  
4. **`phase4_v2_merge.py`**（待实现）→ `phase4_v2_merged/`：concat `phase4_v2` + `phase4_v2_game` 的 `X/y_*`；**pooled 默认不并入 merged**，仅在 manifest 记路径供增广分析  

---

### 3.6 标定轮间 FT 时序（D6=B · 已确认）

| 行为 | 说明 |
|------|------|
| 轮间引导 | 占满 `cal_round_gap_s`；操作者 `v2_guidance_confirm` |
| FT | **后台线程**跑 RoundController 内微调，**不阻塞**开下一轮 |
| 推理 ckpt | 每轮标定开始用「**上轮 FT 提交完成时**」的 ckpt；若仍在跑则用「上轮结束 ckpt」并 emit `ft_lagged` |
| 试点注意 | 首轮无 lag；日志/登记表记 `ft_lagged` 次数供质控 |

---

## 4. 系统架构（当前）

```text
operator.html（v2 模式 · 引导确认 · gate 文本）
    │ WS
orchestrator  phase_mode=v2_session
    └─ SessionRunner.run_v2_session()
          └─ session_v2 → trial_v2 + inference_v2 + adapt_engine
                └─ v2_bridge → scene.__v2scene
    │ 结束
phase4_v2 + phase4_v2_game + pooled → merge → phase4_v2_merged/（D5=A）
```

细节状态见 [`v2_backlog.md`](v2_backlog.md)。

---

## 5. 分级冒烟（验收不必一次跑满）

| 档位 | 内容 | 时长 | 门槛 |
|------|------|------|------|
| **L0** | degraded（无 LSL/权重）；事件时序 | ~15 min | 开发日常 |
| **L1** | pilot.yaml；1 标定轮 + 1 游戏轮 + 引导确认 | ~25 min | 合成板合并前 |
| **L2** | prod.yaml；≥4 标定 + 准入 + ≥2 游戏 | ~60–75 min | 试点前 |

---

## 6. 分阶段路线（规格层）

| Phase | 目标 |
|-------|------|
| A 后端 | orchestrator、操作台 v2、events 审计 → **L0/L1** |
| B 前端 | touch 或模拟 touch、`?v2demo=game` |
| C 操作台 | gate 曲线图、ERD、弱 MI 状态 |
| D 离线 | 双 phase4 + merge + openbmi 同构 |
| E 试点 | sub04–06 → 20 人 |

---

## 7. 验收清单（对照 §3 契约表）

1. L1：Cue 结束 ↔ `mi_start` 间隔 = `cue_s`（±50ms）
2. 标定无早停时 `mi_dur` ≈ 6s；游戏 `mi_dur` ≤ 6s；Score≥5 早停时 `mi_dur` 可 ≈ 4.2s
3. 游戏轮内 ckpt 不变；轮间 FT 后 ckpt 可追溯
4. M2 回归 ≥99%（mi 锚点）；**v2.1 回放计分**与在线事件 `score` 一致
5. `phase4_v2` + `phase4_v2_game` 可合并或分目录有 manifest
6. **无效 trial 不进 FT**；连 5 无效触发降级路径可演练（L1）

---

## 8. 文档层级

1. **本文 + `v2_session.yaml`**
2. `v2_session_mode_spec.md` ✅ 已同步（2026-08-24：D8、第一人称、mi 锚点、touch）
3. `资料/项目计划/采集流程_20被试底座v2.md` ✅ 已同步（2026-08-24：D8、score_reach/touch、三管道+merge）
4. `v2_ui_refit_plan_A.md` ✅ 四档升物段落作废并改写（2026-08-24）
5. `v2_scene_spec.md` ✅ 重写为第一人称 + D8（2026-08-24）
6. `游戏方案_定稿_v3_20260824.md` — v2 标准定稿（含比赛条文对齐清单与修复清单）
7. 已对齐（2026-08-24）：`资料/项目计划/游戏v2开发计划.md`、`资料/项目计划/面向少样本个体适配的运动想象脑机接口系统设计.md`、`资料/比赛要求/基于运动想象的脑-机交互系统技(10).md`（提交技术方案，改写前备份同目录）

---

## 9. 决策寄存器（v1.5）

| # | 决策 |
|---|------|
| D1 | A — touch 后立即 `mi_end` |
| D2 | B — 游戏试次全部入轮间 FT |
| D3 | C — 游戏短片段 fragment 拼接池 |
| D4 | A — 先操作台模拟 touch |
| D5 | A — 自动三管道 + `phase4_v2_merged/` |
| D6 | B — 标定轮间异步 FT + `ft_lagged` |
| D7 | A — weak_mi 照常 `game_rounds` |
| **D8** | **v2.1 在线判定** — 0.6s 栅格；t≤2.4 半票；Score≥5 早停；错类≥5 / Score≤3 无效；连 5 无效熔断→FT(可选)→v1（§3.4） |

---

## 10. 与实验 25

| 实验 25 | v2 在线 |
|---------|---------|
| Stieger 增量 FT 爬坡 | `quiz_curve.csv` / QuizStore |
| k=12/24/… | 准入 gate=60% |

---

## 11. 下一步（实现序）

1. ~~**D8** `v2_session.yaml` + `scoring_v21` + `trial_v2` / `session_v2` 熔断降级~~ ✅
2. 操作台 **模拟 touch** + trial 按 D1 提前 `mi_end`
3. `phase4_v2_game_pooled.py` + `phase4_v2_merge.py`（D3/C + D5）
4. `session_v2` 标定轮间 **异步 FT** + `ft_lagged`（D6）
5. `quiz_curve.csv` 导出
6. L1 合成板冒烟（pilot.yaml + v2.1 计分）
7. sub04 实机

---

*v1.5：D8 在线判定 v2.1 冻结（加权分早停、半票段、连 5 无效熔断）。v1.4：D5–D7 封板。*
