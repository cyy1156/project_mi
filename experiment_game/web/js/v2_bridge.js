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

/** 场景模式下的非遮挡文字横幅（cue/MI 阶段指导语）。pointer-events:none 不挡交互。 */
let _bannerEl = null;
function showBanner(text, sub = "") {
  if (!text) { clearBanner(); return; }
  if (!_bannerEl) {
    _bannerEl = document.createElement("div");
    _bannerEl.id = "v2-stage-banner";
    _bannerEl.style.cssText =
      "position:fixed;left:50%;top:9%;transform:translateX(-50%);text-align:center;" +
      "color:#e8eefc;font-size:40px;font-weight:600;text-shadow:0 2px 8px #000a;" +
      "pointer-events:none;z-index:40;max-width:80vw";
    const subEl = document.createElement("div");
    subEl.className = "banner-sub";
    subEl.style.cssText = "font-size:20px;font-weight:400;opacity:.85;margin-top:10px";
    _bannerEl.appendChild(document.createTextNode(""));
    _bannerEl.appendChild(subEl);
    document.body.appendChild(_bannerEl);
  }
  _bannerEl.firstChild.textContent = text;
  _bannerEl.querySelector(".banner-sub").textContent = sub;
}
function clearBanner() { _bannerEl?.remove(); _bannerEl = null; }

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
  const phase = document.getElementById("phase-tag");
  const t = title || "";
  if (text) {
    // cue 阶段：金色徽章，避免被换行/叠层「吃掉」英文 cue
    const m = String(t).match(/^cue\s*[·\-–—|｜]?\s*([\s\S]*)$/i);
    if (m) {
      text.replaceChildren();
      const badge = document.createElement("span");
      badge.className = "cue-badge";
      badge.textContent = "cue";
      text.appendChild(badge);
      const rest = (m[1] || "").trim();
      if (rest) text.appendChild(document.createTextNode(rest));
      if (phase) phase.textContent = "CUE";
    } else {
      text.textContent = t;
      if (phase && String(phase.textContent).toUpperCase() === "CUE") {
        phase.textContent = "";
      }
    }
  }
  if (subEl) subEl.textContent = sub || "";
  if (cross) cross.classList.toggle("hidden", !showCross);
}

/** 与图三同构：仅 HUD 双层深色圆角框，不用 showBanner 浮层（避免叠字/断行）。 */
function showPromptBoxes(title, sub = "", { cross = false } = {}) {
  clearBanner();
  setHud(title || "", sub || "", Boolean(cross));
}

function miSubtext(label) {
  return MI_GUIDANCE[label] ?? "请按提示想象";
}

function cueText(label, rich = true) {
  const map = rich ? CUE : CUE_PLAIN;
  return map[label] ?? (rich ? "—" : "请按提示想象");
}

function cueTitle(label, data) {
  // 固定以 "cue" 开头；正文与 MI 主框相同（或后端长句）
  const lab = Number(label);
  const plain =
    CUE_PLAIN[lab] ??
    CUE_PLAIN[label] ??
    "请按提示想象";
  let body = plain;
  if (data?.cue_text != null && String(data.cue_text).trim()) {
    body = String(data.cue_text).trim().replace(/^cue\s*[·\-–—|｜]?\s*/i, "");
  }
  return `cue ${body}`;
}

function miTitle(label, data) {
  // 图三主框：短标题「想象：左/右手握紧杯子」；游戏测试长句仅作副文案时仍用短标题
  if (label === 0) return CUE_PLAIN[0];
  if (label === 1 || label === 2) return CUE_PLAIN[label];
  return data?.cue_text || "持续想象";
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
  const s = S(); domClear(); clearBanner();
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
    case "inter_trial_rest": {
      const restDur = data?.duration_s != null ? `（${Number(data.duration_s).toFixed(0)} 秒）` : "";
      const restText = data?.rest_text || "保持静息";
      s ? s.fixation() : domRender("🌙 静息");
      // 与图三同构：双框提示，不叠 showBanner / 不强制十字抢视觉
      showPromptBoxes(`${restText}${restDur}`, MI_GUIDANCE[0], { cross: false });
      break;
    }
    case "prep":
      s ? s.fixation() : domRender("➕");
      showPromptBoxes("", "注视十字，保持放松", { cross: true });
      break;
    case "cue": {
      const title = cueTitle(label, data);
      const sub = data?.cue_sub || miSubtext(label);
      s ? s.v2Cue(label) : domRender(`${CUE[label] ?? ""}`);
      showPromptBoxes(title, sub, { cross: false });
      break;
    }
    case "mi": {
      const title = miTitle(label, data);
      const sub = miSubtext(label);
      if (armFeedbackEnabled(mode, data) && Number(label) !== 0) {
        s ? s.gameLevel(0, false) : domRender(`${CUE[label] ?? "持续想象"}`);
      } else if (armFeedbackEnabled(mode, data) && Number(label) === 0) {
        s ? s.fixation() : domRender(CUE[0]);
      } else {
        // 探针/无伸手反馈：保留场景，提示用图三双框
        s ? s.fixation() : domRender(`${CUE[label] ?? "持续想象"}`);
      }
      showPromptBoxes(title, sub, { cross: false });
      break;
    }
    case "judge": {
      if (data?.signal_bad) {
        showPromptBoxes(cueText(label, false), "请保持放松，稍候继续", { cross: false });
        break;
      }
      if (armFeedbackEnabled(mode, data)) {
        applyArmFeedback(s, data, label);
        showPromptBoxes(
          cueText(label, false),
          data?.cup_grasp ? "拿到了" : miSubtext(label),
          { cross: false },
        );
      } else {
        showPromptBoxes(cueText(label, false), miSubtext(label), { cross: false });
      }
      break;
    }
    case "iti":
    case "trial_end":
      s ? s.iti() : domRender("😌");
      setHud("休息", "准备下一次试次");
      break;
    case "game_end": {
      const sc = Number(data?.score ?? 0);
      const mx = Number(data?.score_max ?? 0);
      const txt = mx > 0 ? `得分 ${sc} / ${mx}` : `得分 ${sc}`;
      s ? s.idle({ title: "游戏结束", sub: txt }) : domRender(`🏁 游戏结束<br/><small style="font-size:22px">${txt}</small>`);
      setHud("游戏结束", txt);
      break;
    }
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
