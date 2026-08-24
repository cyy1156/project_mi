/** v2_bridge · A 案改造核心：{type:"v2_stage"} 消息 → 场景挂点（缺失时 DOM 兜底）。
 *  场景挂点（scene.js 后续实现，注册到 window.__v2scene 即接管渲染）：
 *    cue(label) · fixation() · calProgress(p0..1) · gameLevel(n, reach) · iti() · idle(t)
 *  演示模式：index.html?v2demo=calibration|game（无需后端）。 */
import { WsClient } from "./ws_client.js?v=20260821a";

const S = () => window.__v2scene || null;
const $c = () => document.querySelector("#stage") || document.body;

function domRender(html) { const host = $c(); host.insertAdjacentHTML("beforeend",
  `<div id="v2ov" style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;background:#0b1220ee;color:#e8eefc;font-size:34px;z-index:99"></div>`);
  document.getElementById("v2ov").innerHTML = html; }
function domClear() { document.getElementById("v2ov")?.remove(); }

const CUE = { 1: "🎾🤚 想象：左手抓握", 2: "✍️📝 想象：右手书写", 0: "🌙 保持静息" };

export function handleV2Stage(stage, ctx, data) {
  const s = S(); domClear();
  const label = ctx?.label;
  switch (stage) {
    case "guidance_begin": s ? s.idle("动觉引导") : domRender(`🙌 动觉引导（第 ${data?.round ?? "?"} 轮）<br/><small style="font-size:18px;opacity:.6">操作者抬臂 → 记住感觉 → 睁眼想象复现</small>`); break;
    case "guidance_end": case "round_end": s ? s.idle("") : domRender("✔"); break;
    case "round_start": s ? s.idle(data?.mode === "game" ? "游戏环节" : "标定环节")
                        : domRender(data?.mode === "game" ? "🎮 游戏环节" : "📐 标定环节"); break;
    case "prep": s ? s.fixation() : domRender("➕"); break;
    case "cue": s ? s.cue(label) : domRender(CUE[label] ?? ""); break;
    case "mi": s ? s.calProgress(0) : domRender(`${CUE[label] ?? "持续想象"}<br/><small style="font-size:16px;opacity:.6">请持续想象…</small>`); break;
    case "score_reach": s ? s.gameLevel(4, true) : domRender(`🎯 得分达标 (${data?.score ?? "?"})`); break;
    case "touch": s ? s.gameLevel(4, true) : domRender("🤚 触碰物体"); break;
    case "arm_level": s ? s.gameLevel(data?.level ?? 0, false) : domRender(`⬆ 档位 ${data?.level}/4`); break;
    case "reach": s ? s.gameLevel(4, true) : domRender("🎯 已达标"); break;
    case "iti": case "trial_end": s ? s.iti() : domRender("😌"); break;
    default: break;
  }
}

export function wireV2(client) { /* 由 main.js 分发调用 */ }

/** 演示模式：URL ?v2demo=calibration|game 时启动（不连 ws）。 */
export function maybeDemo() {
  const q = new URLSearchParams(location.search).get("v2demo");
  if (!q) return false;
  const evts = [{ stage: "guidance_begin", data: { round: 1 } }, { stage: "guidance_end" },
                { stage: "round_start", data: { mode: q } }];
  (q === "game" ? [1, 2, 0, 1] : [1, 2, 0]).forEach((label, i) => {
    evts.push({ stage: "trial_start", ctx: { label } }, { stage: "prep" },
      { stage: "cue", ctx: { label } }, { stage: "mi", ctx: { label } });
    if (q === "game") {
      const ticks = [0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2];
      ticks.forEach((t) => evts.push({ stage: "judge", data: { t_rel: t, score: t <= 3 ? t * 0.5 : t - 1.5 } }));
      evts.push({ stage: "score_reach", data: { t_rel: 4.2, score: 5 } });
    }
    evts.push({ stage: "iti" }, { stage: "trial_end", ctx: { label }, data: { summary: { correct: i % 2 === 0 } } });
  });
  evts.push({ stage: "round_end" });
  const gap = (st) => ({ guidance_begin: 2400, guidance_end: 800, round_start: 1000, cue: 1500,
    mi: 4000, judge: 200, score_reach: 1100, touch: 1100, arm_level: 600, reach: 1100, iti: 1200, trial_end: 1000 }[st] ?? 800);
  let i = 0; (function next() { if (i >= evts.length) return domClear();
    const e = evts[i++]; handleV2Stage(e.stage, e.ctx, e.data); setTimeout(next, gap(e.stage)); })();
  return true;
}
