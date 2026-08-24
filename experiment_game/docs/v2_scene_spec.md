# v2 前端场景规格（M4 · 第一人称伸手/触碰 + D8 计分反馈）

> 状态：**2026-08-24 按 v2 标准重写**（旧版"第三人称机械臂 + 4 档位移"作废，见 `v2_upgrade_plan.md` v1.5 §3.1）。后端（trial_v2 / inference_v2 / adapt_engine）已就绪；前端按 [refit 方案 A](v2_ui_refit_plan_A.md) 在 v1 页面原地扩展。
> 规格源：`v2_upgrade_plan.md` v1.5 + `config/v2_session.yaml`；本文只定义前端新增部分。

## 1. 两个场景（均为第一人称，复用 v1 世界）

### 1.1 标定场景（calibration）
- 元素：注视点 → Cue 动画（左=`anim_ball_grasp` 网球抓握 / 右=`anim_writing` 书写 / 静息=`icon_rest` 固定图标）→ 想象期底部进度条（t=0..6s 推进，**无分类反馈**）→ ITI 黑屏。
- 素材：按比赛 2.1——左手=抓握**网球**（当前为抓杯，待换）、右手=执笔书写往复（当前为通用伸手，待加笔）、静息图标。后端 `cue_kind`（`trial_v2.py` CUE_KIND）已冻结，前端必须消费。

### 1.2 游戏场景（game）
- 第一人称：左/右 Cue → **同侧手**伸向并抓握桌面目标物（复用现有手部骨架与抓取动画）。
- 反馈驱动（D8）：每档 `judge` 消息渲染**得分进度**（当前 Score / 目标 5.0，含半票档标识）；`score_reach` → 抓取成功动画 + 提前进 ITI；`touch`（模拟/碰撞）→ 立即抓取定格 + ITI；`trial_invalid_*` → 中性失败提示（不渲染手部抓取）。
- **静息试次**：无手部动画，显示静息指示（图标/文案）；**禁止**任何抓取动画触发。
- 后端保证：早停后脑电录至真实 `mi_end`，前端展示无需等待满 6s。

## 2. 阶段通知（后端 on_stage → ws → 前端）

| stage | data | 前端动作 |
|-------|------|----------|
| `trial_start` | ctx | 重置场景 |
| `prep` | — | 注视点 |
| `cue` | ctx(label, cue_kind) | 播对应 Cue 动画（2s） |
| `mi` | — | 标定=进度条推进；游戏=手部就绪 |
| `judge` | {t_rel, pred, p_max, gated, weight, score} | 游戏场景得分进度更新（**当前前端缺此 case，见代码审查 F11**） |
| `score_reach` | {t_rel, score} | 抓取成功动画 → ITI |
| `touch` | {t_rel} | 触碰定格 → ITI |
| `trial_invalid` | {reason, score} | 中性失败提示 |
| `iti` | — | 黑屏 |
| `trial_end` | {summary} | 短暂结果页（游戏场景） |
| `guidance_begin/end` | {round, passed} | 引导页（无 EEG 显示，操作员确认按钮） |

（旧事件名 `arm_level`/`reach` 作废；`arm_level` 可保留至旧代码清理完成。）

## 3. 操作台 v2 增量（M5 对接）

- 轮控制：`round_start(round, mode)` / `round_end`；轮间显示**累积小考曲线**与**准入状态**（pass/extend/weak_mi）。
- **模拟 touch 按钮**（D4 先行）：注入 `touch` 客户端事件 → `session_v2` 置 `touch_pending` → trial 立即 `mi_end`（三层入口当前均缺，见代码审查 F12）。
- ERD 反馈面板：轮间计算对侧 ERD + laterality（复用 H 门控公式），操作员可见。
- 注意力分数：逐试次日志字段（后端写入 events，操作台仅展示）。

## 4. 验收

1. 合成板全流程：引导 → 标定轮 → 准入 → 游戏轮，无事件丢失（events 完整含 `judge/score_reach/touch/trial_invalid_*`）。
2. cue 呈现与 `cue` 事件时戳差 <50ms（复用 Phase0–1 对齐校验）。
3. 早停（score_reach/touch）时：前端立即进 ITI；`mi_end` reason 与展示一致。
4. `?v2demo=calibration|game` 渲染循环正常（当前黑屏，见代码审查 F11/M6）。
5. 静息试次无任何抓取动画；rest `score_reach` 不触发手部动作。
