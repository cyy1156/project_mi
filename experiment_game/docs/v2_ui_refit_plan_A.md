# v2 前端改造方案 · A 案（保持原有第一人称世界，升级到 v2 契约）

> 2026-08-23 · 决策：弃用临时页（v2_subject.html / v2_main.js / v2_arm_scene.js → 删除），
> **在 v1 index.html + scene.js + main.js 上原地扩展**。  
> **2026-08-24 对齐 v2 标准**：本文 §2「四档升物」玩法作废（`v2_upgrade_plan.md` v1.5 §3.1），改为 **D8 加权计分驱动的伸手/触碰反馈**；stage 契约中 `arm_level/reach` 由 `judge/score_reach/touch` 取代。页面删除与原地扩展部分已执行。

## 1. 保持不动的部分（原有逻辑全保留）

| 组件 | 说明 |
|------|------|
| scene.js 世界 | 程序化贴图（木纹/墙/地板/地毯）、PMREM 环境反射、三场景（家庭/医院/学校） |
| 第一人称手部 | 解剖结构手（胶囊+关节球、五指弯曲）、前臂+袖口；**抓握动画直接当左手 Cue** |
| main.js / ws_client.js | 连接、ready/sync、既有消息分支原样 |
| orchestrator / 后端 | 不改（v2 会话模式接线是另一件独立事项） |

## 2. 新增：v2_stage 消息分支（一个新文件 + scene.js 少量挂点）

**新文件 `web/js/v2_bridge.js`（~150 行）**——接收 `{type:"v2_stage", stage, ctx, data}`，翻译成对 scene.js 的调用；内置演示播放器（同契约，`index.html?v2demo=calibration|game` 可无后端验证）。

**scene.js 新增挂点（4 个公开函数，尽量薄）**：

| 挂点 | 行为 | 复用/新增 |
|------|------|-----------|
| `v2Cue(label)` | 左=现有五指抓握动画（对网球物）；右=右手+笔的书写往复（笔=小胶囊挂指尖）；0=双手自然放腿上 | 抓握**复用**；书写/休息**新增**（同套手部骨架） |
| `v2Fixation()` | 相机回正 + 注视点（现有 UI 层） | 复用 |
| `v2CalProgress(t/6s)` | 底部细进度条叠加层（无分类反馈） | 新增（DOM 层） |
| `v2GameFeedback(score, reach)` | **D8 得分驱动的伸手反馈**：每档 `judge` → 手部向目标物推进 + 得分进度更新（Score/5.0）；`score_reach`/`touch` → 手完成抓取（物体发绿光）→ ITI | 得分推进**新增**；抓取**复用**手部动画 |

**玩法语义（A 案核心 · 2026-08-24 更新）**：游戏协同是**在同一第一人称世界里"同侧手伸够桌面目标物"**——D8 加权计分（0.6s 栅格 ×10 档，≤2.4s 半票）驱动手部推进与得分可视化；**Score≥5（score_reach）或 touch（模拟/碰撞）即抓住目标、试次提前结束**（`mi_end` 即停，短试次离线走 fragment 池，由后端保证）。旧"四档升物"作废。诱导逻辑与 v1 完全连续。

## 3. main.js 改动（一处）

消息分发处加一行：`else if (msg.type === "v2_stage") v2Bridge.handle(msg)`（import v2_bridge.js）。v1 旧消息路径零改动。

## 4. 删除清单

`web/v2_subject.html`、`web/js/v2_main.js`、`web/js/v2_arm_scene.js`（git 历史可找回）。

## 5. 契约对照（不变，重申）

stage：`guidance_begin/end · round_start(mode) · prep · cue(label) · mi · judge(t_rel/score/weight) · score_reach · touch · trial_invalid · iti · trial_end(summary) · round_end`
（`arm_level/reach` 作废；`judge` 携带每档得分供前端进度渲染——当前 `v2_bridge.js` 缺 `judge` case，见代码审查 F11。）

## 6. 实施顺序与验收

1. ✅ 删临时三文件 → 2. 🟡 scene.js 挂点（基础已挂；书写动画、网球资产、D8 得分推进、静息指示待补）→ 3. ✅ v2_bridge.js + main.js 一行（`judge`/`score_reach`/`touch` 分支与 `?v2demo` 渲染循环待修）→ 4. 模拟 touch 三层入口（操作台按钮 → WS → `touch_pending`，D4）→ 5. `index.html?v2demo=game` 视觉验收 → 6. 合成板接 ws 实测一完整会话（cue 时戳差 <50ms，复用 Phase0–1 校验）。

> 实现时对照 scene.js 既有导出（换物/场景切换/手部动画入口）命名挂点，不重写已有函数。
