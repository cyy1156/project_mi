/** v2_bridge · {type:"v2_stage"} → 场景挂点（缺失时 DOM 兜底）。
 *  挂点：cue · fixation · gameLevel · iti · idle
 *  arm_reach：每窗判对 +1 驱动伸手；≥5 拿杯（MI 仍跑完全程）
 *  演示：index.html?v2demo=calibration|game */
import { WsClient } from "./ws_client.js?v=20260825arm1";

const S = () => window.__v2scene || null;
const $c = () => document.querySelector("#stage") || document.body;

function domRender(html) { const host = $c(); host.insertAdjacentHTML("beforeend",
  `<div id="v2ov" style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;background:#0b1220ee;color:#e8eefc;font-size:34px;z-index:99"></div>`);
  document.getElementById("v2ov").innerHTML = html; }
function domClear() { document.getElementById("v2ov")?.remove(); }

const MI_GUIDANCE = {
  1: "请想象左手正在握紧桌上的杯子，感受手指收紧、前臂用力，不要真的动。",
  2: "请想象右手正在握紧桌上的杯子，感受手指收紧、前臂用力，不要真的动。",
  0: "身体放松，什么都不要想，保持安静。",
};
const CUE = { 1: "🎾🤚 想象：左手握紧杯子", 2: "🎾🤚 想象：右手握紧杯子", 0: "🌙 保持静息" };
const CUE_PLAIN = { 1: "想象：左手握紧杯子", 2: "想象：右手握紧杯子", 0: "保持静息" };

let subjectFeedbackMode = "none";

export function setSubjectFeedbackMode(mode) {
  subjectFeedbackMode = mode === "arm_reach" ? "arm_reach" : "none";
}

function armFeedbackEnabled(mode, data) {
  if (subjectFeedbackMode !== "arm_reach") return false;
  const m = mode || data?.mode;
  return m === "game" || m === "probe";
}

function setHud(title, sub = "", showCross = false) {
  const text = document.getElementById("hud-text");
  const subEl = document.getElementById("hud-sub");
  const cross = document.getElementById("cross");
  if (text) text.textContent = title || "";
  if (subEl) subEl.textContent = sub || "";
  if (cross) cross.classList.toggle("hidden", !showCross);
}

function miSubtext(label) {
  return MI_GUIDANCE[label] ?? "请按提示想象";
}

function cueText(label, rich = true) {
  const map = rich ? CUE : CUE_PLAIN;
  return map[label] ?? (rich ? "—" : "请按提示想象");
}

function resolveRoundNo(ctx, data) {
  const r = data?.round ?? data?.block ?? ctx?.round;
  return r != null && r !== "" ? r : "?";
}

function guidanceIdle(data, ctx) {
  const round = data?.round ?? ctx?.round;
  const isPhase0 = round === 0 || round == null;
  return {
    title: isPhase0 ? "动觉引导" : `动觉引导 · 第 ${round} 轮`,
    sub: "两手分别抓握杯子 → 记住抓握动作 → 睁眼按指导语想象复现",
  };
}

function roundIdle(data, mode, ctx) {
  const isGame =
    mode === "game" ||
    data?.mode === "game" ||
    (subjectFeedbackMode === "arm_reach" && mode === "probe");
  const round = resolveRoundNo(ctx, data);
  return {
    title: isGame ? "游戏环节" : "标定环节",
    sub: `第 ${round} 轮 · ${isGame ? "请按提示想象" : "请按提示完成标定试次"}`,
  };
}

function isGameMode(mode, data) {
  return mode === "game" || data?.mode === "game";
}

function applyArmFeedback(s, data, label) {
  if (Number(label) === 0) {
    s ? s.fixation() : null;
    return;
  }
  if (data?.signal_bad) return;
  const level = data?.arm_level != null ? Number(data.arm_level) : null;
  const grasp = !!data?.cup_grasp;
  if (level == null && !grasp) return;
  if (s) s.gameLevel(grasp ? 4 : level, grasp);
  else if (grasp) domRender("🎯 拿到了");
  else domRender("⬆");
}

export function handleV2Stage(stage, ctx, data) {
  const s = S(); domClear();
  const label = ctx?.label;
  const mode = ctx?.mode || data?.mode;
  const game = isGameMode(mode, data);
  switch (stage) {
    case "guidance_begin": {
      const idle = guidanceIdle(data, ctx);
      s ? s.idle(idle) : domRender(`🙌 ${idle.title}<br/><small style="font-size:18px;opacity:.6">${idle.sub}</small>`);
      setHud(idle.title, idle.sub);
      break;
    }
    case "guidance_end":
      s ? s.idle({ title: "引导完成", sub: data?.inter_round ? "准备进入下一轮" : "准备进入标定" }) : domRender("✔");
      setHud("引导完成", data?.inter_round ? "准备进入下一轮" : "准备进入标定");
      break;
    case "round_end": {
      const idle = roundIdle(data, mode, ctx);
      const round = resolveRoundNo(ctx, data);
      s ? s.idle({ title: `${idle.title}结束`, sub: `第 ${round} 轮已完成` }) : domRender("✔");
      setHud(`${idle.title}结束`, `第 ${round} 轮已完成`);
      break;
    }
    case "gate_pass":
      s ? s.idle({ title: "准入建议：已过线", sub: "请等待操作员确认后进入游戏" }) : domRender('✅ 准入建议：已过线<br/><small style="font-size:18px;opacity:.6">等待操作员确认</small>');
      setHud("准入建议：已过线", "请等待操作员确认");
      break;
    case "weak_mi":
      s ? s.idle({ title: "弱 MI 建议", sub: "未达门槛 · 等待操作员确认是否进入游戏" }) : domRender("⚠ weak_mi · 等待操作员确认");
      setHud("弱 MI 建议", "等待操作员确认是否进入游戏");
      break;
    case "round_start": {
      const idle = roundIdle(data, mode, ctx);
      s ? s.idle(idle) : domRender(idle.title === "游戏环节" ? "🎮 游戏环节" : "📐 标定环节");
      setHud(idle.title, idle.sub);
      break;
    }
    case "rest_start":
    case "inter_trial_rest":
      s ? s.fixation() : domRender("🌙 静息");
      setHud("静息", MI_GUIDANCE[0], true);
      break;
    case "prep":
      s ? s.fixation() : domRender("➕");
      setHud("", "注视十字，保持放松", true);
      break;
    case "cue":
      s ? s.cue(label) : domRender(`${CUE[label] ?? ""}<br/><small style="font-size:16px;opacity:.75">${miSubtext(label)}</small>`);
      setHud(cueText(label, false), miSubtext(label));
      break;
    case "mi":
      if (armFeedbackEnabled(mode, data) && Number(label) !== 0) {
        s ? s.gameLevel(0, false) : domRender(`${CUE[label] ?? "持续想象"}`);
        setHud(cueText(label, false), miSubtext(label));
      } else if (armFeedbackEnabled(mode, data) && Number(label) === 0) {
        s ? s.fixation() : domRender(CUE[0]);
        setHud(cueText(0, false), miSubtext(0));
      } else {
        s ? s.fixation() : domRender(
          `${CUE[label] ?? "持续想象"}<br/><small style="font-size:16px;opacity:.75">${miSubtext(label)}</small>`
        );
        setHud(cueText(label, false), miSubtext(label));
      }
      break;
    case "judge": {
      if (data?.signal_bad) {
        setHud(cueText(label, false), "请保持放松，稍候继续");
        break;
      }
      if (armFeedbackEnabled(mode, data)) {
        applyArmFeedback(s, data, label);
        setHud(cueText(label, false), data?.cup_grasp ? "拿到了" : miSubtext(label));
      } else {
        setHud(cueText(label, false), miSubtext(label));
      }
      break;
    }
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

/** 演示模式：URL ?v2demo=calibration|game（仅 Left/Right，含 Cue前 Rest） */
export function maybeDemo() {
  const q = new URLSearchParams(location.search).get("v2demo");
  if (!q) return false;
  setSubjectFeedbackMode(q === "game" ? "arm_reach" : "none");
  const evts = [{ stage: "guidance_begin", data: { round: 1 } }, { stage: "guidance_end" },
                { stage: "round_start", data: { mode: q } }];
  const labels = q === "game" ? [1, 2, 1, 2] : [1, 2, 1];
  labels.forEach((label) => {
    const mode = q;
    evts.push(
      { stage: "inter_trial_rest", ctx: { label, mode }, data: { duration_s: 4 } },
      { stage: "trial_start", ctx: { label, mode } },
      { stage: "prep", ctx: { label, mode } },
      { stage: "cue", ctx: { label, mode } },
      { stage: "mi", ctx: { label, mode } },
    );
    if (q === "game") {
      [1, 2, 3, 4, 5].forEach((score, j) => evts.push({
        stage: "judge",
        ctx: { label, mode },
        data: { score, arm_level: Math.min(3, j), cup_grasp: score >= 5 },
      }));
    }
    evts.push({ stage: "iti", ctx: { label, mode } }, { stage: "trial_end", ctx: { label, mode } });
  });
  evts.push({ stage: "round_end", data: { mode: q, round: 1 } });
  const gap = (st) => ({ guidance_begin: 2400, guidance_end: 800, round_start: 1000,
    inter_trial_rest: 1200, cue: 1500, mi: 1200, judge: 450, iti: 1200, trial_end: 1000 }[st] ?? 800);
  let i = 0; (function next() { if (i >= evts.length) return domClear();
    const e = evts[i++]; handleV2Stage(e.stage, e.ctx, e.data); setTimeout(next, gap(e.stage)); })();
  return true;
}
