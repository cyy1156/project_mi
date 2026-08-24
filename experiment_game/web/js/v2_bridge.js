/** v2_bridge · A 案改造核心：{type:"v2_stage"} 消息 → 场景挂点（缺失时 DOM 兜底）。
 *  场景挂点（scene.js 后续实现，注册到 window.__v2scene 即接管渲染）：
 *    cue(label) · fixation() · calProgress(p0..1) · gameLevel(n, reach) · iti() · idle(t)
 *  演示模式：index.html?v2demo=calibration|game（无需后端）。 */
import { WsClient } from "./ws_client.js?v=20260824prompt3";

const S = () => window.__v2scene || null;
const $c = () => document.querySelector("#stage") || document.body;

function domRender(html) { const host = $c(); host.insertAdjacentHTML("beforeend",
  `<div id="v2ov" style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;background:#0b1220ee;color:#e8eefc;font-size:34px;z-index:99"></div>`);
  document.getElementById("v2ov").innerHTML = html; }
function domClear() { document.getElementById("v2ov")?.remove(); }

const CUE = { 1: "🎾🤚 想象：左手抓握", 2: "✍️📝 想象：右手书写", 0: "🌙 保持静息" };
const CUE_PLAIN = { 1: "想象：左手抓握", 2: "想象：右手书写", 0: "保持静息" };

function setHud(title, sub = "", showCross = false) {
  const text = document.getElementById("hud-text");
  const subEl = document.getElementById("hud-sub");
  const cross = document.getElementById("cross");
  if (text) text.textContent = title || "";
  if (subEl) subEl.textContent = sub || "";
  if (cross) cross.classList.toggle("hidden", !showCross);
}

function cueText(label, rich = true) {
  const map = rich ? CUE : CUE_PLAIN;
  return map[label] ?? (rich ? "—" : "请按提示想象");
}

function guidanceIdle(data) {
  const round = data?.round;
  const isPhase0 = round === 0 || round == null;
  return {
    title: isPhase0 ? "动觉引导" : `动觉引导 · 第 ${round} 轮`,
    sub: "操作者抬臂 → 记住感觉 → 睁眼想象复现",
  };
}

function roundIdle(data, mode) {
  const isGame = mode === "game" || data?.mode === "game";
  const round = data?.round ?? "?";
  return {
    title: isGame ? "游戏环节" : "标定环节",
    sub: `第 ${round} 轮 · ${isGame ? "请按提示想象并得分" : "请按提示完成标定试次"}`,
  };
}

export function handleV2Stage(stage, ctx, data) {
  const s = S(); domClear();
  const label = ctx?.label;
  const mode = ctx?.mode || data?.mode;
  switch (stage) {
    case "guidance_begin": {
      const idle = guidanceIdle(data);
      s ? s.idle(idle) : domRender(`🙌 ${idle.title}<br/><small style="font-size:18px;opacity:.6">${idle.sub}</small>`);
      setHud(idle.title, idle.sub);
      break;
    }
    case "guidance_end":
      s ? s.idle({ title: "引导完成", sub: data?.inter_round ? "准备进入下一轮" : "准备进入标定" }) : domRender("✔");
      setHud("引导完成", data?.inter_round ? "准备进入下一轮" : "准备进入标定");
      break;
    case "round_end": {
      const idle = roundIdle(data, mode);
      s ? s.idle({ title: `${idle.title}结束`, sub: `第 ${data?.round ?? "?"} 轮已完成` }) : domRender("✔");
      setHud(`${idle.title}结束`, `第 ${data?.round ?? "?"} 轮已完成`);
      break;
    }
    case "gate_pass":
      s ? s.idle({ title: "准入通过", sub: "即将进入游戏环节，请准备" }) : domRender('✅ 准入通过<br/><small style="font-size:18px;opacity:.6">进入游戏</small>');
      setHud("准入通过", "即将进入游戏环节");
      break;
    case "weak_mi":
      s ? s.idle({ title: "弱 MI 标记", sub: "标定未完全达标，仍将进入游戏（全程标记 weak_mi）" }) : domRender("⚠ weak_mi · 继续游戏");
      setHud("弱 MI 标记", "仍将进入游戏环节");
      break;
    case "round_start": {
      const idle = roundIdle(data, mode);
      s ? s.idle(idle) : domRender(idle.title === "游戏环节" ? "🎮 游戏环节" : "📐 标定环节");
      setHud(idle.title, idle.sub);
      break;
    }
    case "prep":
      s ? s.fixation() : domRender("➕");
      setHud("", "注视十字，保持放松", true);
      break;
    case "cue":
      s ? s.cue(label) : domRender(CUE[label] ?? "");
      setHud(cueText(label, false), "记住提示，即将开始想象");
      break;
    case "mi":
      s ? s.calProgress(0) : domRender(`${CUE[label] ?? "持续想象"}<br/><small style="font-size:16px;opacity:.6">请持续想象…</small>`);
      setHud(cueText(label, false), "请持续想象…");
      break;
    case "judge": {
      const sc = data?.score;
      if (s && sc != null) s.calProgress(Math.min(1, Number(sc) / 5));
      const modeHint = mode === "game" ? "游戏" : "标定";
      setHud(
        cueText(label, false),
        sc != null ? `${modeHint} · Score ${Number(sc).toFixed(1)}` : `${modeHint} · 判定中…`
      );
      break;
    }
    case "score_reach":
      s ? s.gameLevel(4, true) : domRender(`🎯 得分达标 (${data?.score ?? "?"})`);
      setHud("得分达标", `Score ${data?.score ?? "?"}`);
      break;
    case "touch":
      s ? s.gameLevel(4, true) : domRender("🤚 触碰物体");
      setHud("触碰物体", "想象完成");
      break;
    case "arm_level":
      s ? s.gameLevel(data?.level ?? 0, false) : domRender(`⬆ 档位 ${data?.level}/4`);
      setHud(`档位 ${data?.level ?? 0}/4`, "继续想象");
      break;
    case "reach":
      s ? s.gameLevel(4, true) : domRender("🎯 已达标");
      setHud("已达标", "继续想象");
      break;
    case "iti":
    case "trial_end":
      s ? s.iti() : domRender("😌");
      setHud("休息", "准备下一次试次");
      break;
    default:
      break;
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
