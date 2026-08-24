# v2 会话模式 · 完整需求规格（动觉引导→4–6 轮标定→准入→第一人称游戏协同）

> 2026-08-23 初稿 · **2026-08-24 按 v2 标准重写**：规格源以 [`v2_upgrade_plan.md`](v2_upgrade_plan.md) v1.5（D1–D8）+ `config/v2_session.yaml` 为准；本文是实现层需求复述，冲突时以上游为准。  
> 入口：操作台选"v2 会话模式"→ `SessionRunner.run_v2_session()`（与现有 `run_all` 并列，复用事件/标记/LSL/bridge 全套基建）。

## 一、流水线骨架（用户原意 → v2 落实）

1. 正式采集前、不采集脑电时，加入**动觉引导**：操作者抬起被试手臂，被试记住这种感觉，直到能睁眼想象出画面 → 进入实验。
2. **4–6 轮标定**，每轮 **18 试次**；轮间操作者重复动觉引导。
3. 轮间微调：在同一被试上一轮微调后的权重上继续（**异步**，D6），每轮**前 12 试次微调、后 6 试次小考**。
4. 累积小考卷三分类 **≥60%** → 进入激活游戏协同（小考 ≥12 试次起判）。
5. 游戏协同 ×2–3 轮 ×16 试次：Cue 结束后想象 **≤6s**；**D8 加权计分 Score≥5 或 touch → 提前 `mi_end`**（D1/D8）；未触发则满 6s。

## 二、统计/数据修正记录（沿用）

| 原设计 | 修正 | 理由 |
|--------|------|------|
| 16 试次 = 12 FT + 4 小考 | **18 = 12 FT + 6 小考**（每类恰好 2 个进小考） | 4 试次三分类只有 0/25/50/75/100 四档，"≥60%"=要求 75%，门槛被噪声支配 |
| 想象 6s、到位即停、脑电照录满 6s | **`mi_end` 即停录**（touch/`score_reach`），短试次离线走 **fragment 拼接池**（D3） | v1.5 决策：游戏试次不必满 6s；`phase4_v2_game` 切至真实 `mi_end` |

## 三、单试次结构（冻结 · ≤13s）

```text
0–2s   prep（注视）
2–4s   Cue 动画（左=抓网球 / 右=书写 / 静息=图标）；MI 未开始
4s     mi_start
4–10s  MI ≤6s：D8 判定（mi 后 0.6/1.2/…/6.0s 共 10 档；半票段 ≤2.4s）
       标定：恒满 6s（无早停）
       游戏：touch（D1）/ Score≥5（score_reach）/ 错类作废 / 满 6s → mi_end
+3s    ITI
```

## 四、流水线实现（嵌进 SessionRunner）

### 阶段 0 · 动觉引导体验（不采集脑电，5–10 min）
- 操作者抬被试手臂（左=抓握动作、右=书写动作）；被试闭眼感受→睁眼想象复现。
- **通过标准**：被试自评能清晰复现（连续 2 次成功想象），口头确认（操作台 `v2_guidance_confirm`）。
- 事件：`guidance_begin/guidance_end(passed, duration)` 落盘（脑电不采集）。

### 阶段 1 · 标定轮 ×4–6（每轮 18 试次 = 12 FT + 6 小考）
- 试次结构同第三节，标定模式无早停、无分类反馈（进度条）。
- 排程：3 子块 ×6（2L+2R+2Rest 置换，同类连出 ≤2）。
- 轮间（`cal_round_gap_s`）：操作者重复动觉引导 ‖ **后台异步增量微调**（D6：上轮权重继续 + 15% 源域回放，仅用本轮 12 试次窗；未完成记 `ft_lagged`，下轮用上轮结束 ckpt）。
- 后 6 试次进**累积小考卷**（永不进 FT，防泄漏）；每轮结束画爬坡曲线点（k_ft=12/24/36/48…）。

### 阶段 2 · 准入判定（系统常量，现场禁改）
- `gate_enter_three = 0.60`，累积小考卷 ≥12 试次起判（试次级 = D8 10 档栅格多数票）。
- 达标 → 进游戏；未达 → 加轮（最多 6 轮）；6 轮仍未达 → `weak_mi`：用历史最优 ckpt 照常 `game_rounds`（D7），全程标记。

### 阶段 3 · 游戏协同 ×2–3 轮（每轮 16 试次，6/5/5 轮转）
- 模型 = **准入通过时的个人模型**（垃圾反馈防线）。
- D8 判定驱动第一人称反馈：Score≥5 → `score_reach` 提前结束；touch（操作台模拟，D4 → 后续场景碰撞）→ 立即 `mi_end`；错类累计 ≥5 或终局 Score≤3 → 试次无效；连 5 无效 → 熔断（有效窗 ≥6 → 补一轮 FT）→ 降级 v1 记录。
- 轮间：引导复习 + 增量 FT（**全部有效试次窗入训**，D2）+ DriftGuard（待接线）。

## 五、系统常量（`config/v2_session.yaml`，现场禁改）

| 常量 | 值 | 含义 |
|------|-----|------|
| `cal_rounds_min/max` | 4 / 6 | 标定轮数（设计值；小考 ≥12 且 ≥60% 可提前准入） |
| `trials_per_round` | 18 | 12 FT + 6 小考 |
| `gate_enter_three` | 0.60 | 准入门槛 |
| `game_rounds` / `game_trials_per_round` | 2–3 / 16 | 游戏轮配置 |
| `judgment_step_s` / `judgment_half_weight_until_s` | 0.6 / 2.4 | D8 栅格与半票段 |
| `score_early_stop` / `wrong_class_abort` / `score_invalid_max` | 5.0 / 5.0 / 3.0 | D8 早停 / 错类作废 / 无效 |
| `consecutive_invalid_abort` / `ft_min_valid_trials` | 5 / 6 | 熔断 / 熔断后 FT 门槛 |
| `group_lr` / `replay_ratio` / `drift_patience` / `task_p_on` | 1e-4 / 0.15 / 2 / 0.6 | 适配与门控 |

## 六、实现落点（experiment_game 内）

1. `session_runner` / `orchestrator`：`phase_mode="v2_session"` → `run_v2_session()`（已实现）。
2. `trial_v2.py`（已就绪）：试次状态机、D8 计分挂接、标定/游戏排程器、引导阶段。
3. `adapt_engine`（已就绪）：增量微调、累积小考、准入门槛、漂移保护——轮间调用（回放池/DriftGuard 接线待修，见代码审查）。
4. `inference_v2.py`（已就绪）：在线推理（判定点取窗，mi 锚点）——`trial_v2` 的 `judgment_fn` 调它。
5. 前端：v1 页面原地扩展（refit 方案 A）——`v2_bridge.js` + scene.js `__v2scene` 挂点；D8 逐档 `judge` 渲染与模拟 touch 待补（见 `v2_scene_spec.md`）。

## 七、验收

1. 合成板完整会话（引导→标定→准入→游戏），FT 真实运行，事件完整（含 `judge`（每档 `t_rel/score`）/`score_reach`/`touch`/`trial_invalid_*`/`guidance_*`）。
2. cue 时戳与呈现误差 <50ms（复用 Phase0–1 校验）；Cue 结束 ↔ `mi_start` 间隔 = `cue_s`（±50ms）。
3. `score_reach`/`touch` 后立即进 ITI，`mi_end` 事件 reason 正确；早停试次 `mi_dur` 可 ≈4.2s。
4. 累积小考曲线与手算一致；准入判定日志可审计；无效试次不进 FT（断言）。

## 八、与采集流程文档的关系

本规格 = [采集流程 v2.0](../../资料/项目计划/采集流程_20被试底座v2.md) 的**软件实现版**（2026-08-24 已同步 v2 标准）。比赛交付时操作员启动"v2 会话模式"即走完一个被试的完整适配流。
