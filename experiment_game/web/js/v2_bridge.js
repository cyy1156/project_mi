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

/** cue 画面最短保留：防止 cue/mi 挤在同一帧时只画出 MI（用户完全看不到 cue） */
let _cueGate = null; // { until:number, timer:number|null }

function clearCueGate() {
  if (_cueGate?.timer != null) clearTimeout(_cueGate.timer);
  _cueGate = null;
}

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

function setHud(title, sub = "", showCross = false, opts = {}) {
  const text = document.getElementById("hud-text");
  const subEl = document.getElementById("hud-sub");
  const cross = document.getElementById("cross");
  const phase = document.getElementById("phase-tag");
  const t = title || "";
  if (text) {
    if (opts.cueSplit) {
      // 双行白字黑底：上行单独 cue，下行想象短句——不可能漏看
      text.replaceChildren();
      const line1 = document.createElement("div");
      line1.className = "cue-line";
      line1.textContent = "cue";
      const line2 = document.createElement("div");
      line2.className = "cue-body";
      line2.textContent = String(opts.cueBody || t).replace(/^cue\s*[·\-–—|｜]?\s*/i, "");
      text.append(line1, line2);
    } else {
      text.textContent = t;
    }
  }
  if (phase) {
    phase.classList.toggle("phase-cue", Boolean(opts.cueSplit));
    if (opts.cueSplit) phase.textContent = "CUE";
    else if (String(phase.textContent).toUpperCase() === "CUE") phase.textContent = "";
  }
  if (subEl) subEl.textContent = sub || "";
  if (cross) cross.classList.toggle("hidden", !showCross);
}

/** 与图三同构：仅 HUD 双层深色圆角框，不用 showBanner 浮层（避免叠字/断行）。 */
function showPromptBoxes(title, sub = "", { cross = false, cueSplit = false, cueBody = "" } = {}) {
  clearBanner();
  setHud(title || "", sub || "", Boolean(cross), { cueSplit, cueBody });
}

function miSubtext(label) {
  return MI_GUIDANCE[label] ?? "请按提示想象";
}

function cueText(label, rich = true) {
  const map = rich ? CUE : CUE_PLAIN;
  return map[label] ?? (rich ? "—" : "请按提示想象");
}

function cueBodyPlain(label, data) {
  const lab = Number(label);
  const plain =
    CUE_PLAIN[lab] ??
    CUE_PLAIN[label] ??
    "请按提示想象";
  if (data?.cue_text != null && String(data.cue_text).trim()) {
    return String(data.cue_text).trim().replace(/^cue\s*[·\-–—|｜]?\s*/i, "");
  }
  return plain;
}

function cueTitle(label, data) {
  return `cue · ${cueBodyPlain(label, data)}`;
}

function miTitle(label, data) {
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
  const isV3Block = data?.block != null || data?.cond != null;
  const auto = Boolean(data?.auto);
  const graspSub = "两手分别抓握杯子 → 记住抓握动作 → 睁眼按指导语想象复现";
  const autoSub = "合成/仿真：自动确认中…";
  if (isV3Block) {
    const n = data?.block ?? round ?? "?";
    return {
      title: `动觉引导 · 第 ${n} 块`,
      sub: auto ? autoSub : graspSub,
    };
  }
  const isPhase0 = round === 0 || round == null;
  return {
    title: isPhase0 ? "动觉引导" : `动觉引导 · 第 ${round} 轮`,
    sub: auto ? autoSub : graspSub,
  };
}

function roundIdle(data, mode, ctx) {
  const isV3Block = data?.block != null || data?.cond != null;
  if (isV3Block) {
    const n = data?.block ?? data?.round ?? ctx?.round ?? "?";
    const cond = data?.cond ? ` · ${data.cond}` : "";
    return {
      title: `第 ${n} 块${cond}`,
      sub: "准备开始",
    };
  }
  const n = resolveRoundNo(ctx, data);
  const isGame = isGameMode(mode, data);
  return {
    title: isGame ? "游戏环节" : "标定环节",
    sub: `第 ${n} 轮`,
  };
}

function isGameMode(mode, data) {
  const m = mode || data?.mode;
  return m === "game" || m === "probe";
}

function applyArmFeedback(s, data, label) {
  const level = Number(data?.arm_level ?? 0);
  const reach = Boolean(data?.cup_grasp);
  const progress = data?.arm_progress;
  if (s?.v2GameLevel) s.v2GameLevel(level, reach, label, progress);
  else if (s?.gameLevel) s.gameLevel(level, reach, label, progress);
}

function applyMiStage(s, label, mode, data) {
  const title = miTitle(label, data);
  const sub = miSubtext(label);
  if (armFeedbackEnabled(mode, data) && Number(label) !== 0) {
    s ? s.gameLevel(0, false, Number(label), 0) : domRender(title);
  } else if (armFeedbackEnabled(mode, data) && Number(label) === 0) {
    s ? s.fixation() : domRender(title);
  } else {
    s ? s.fixation() : domRender(title);
  }
  showPromptBoxes(title, sub, { cross: false });
}

export function handleV2Stage(stage, ctx, data) {
  const s = S();
  if (stage !== "cue" && stage !== "mi") clearCueGate();

  const label = ctx?.label;
  const mode = ctx?.mode || data?.mode;
  switch (stage) {
    case "guidance_begin": {
      const idle = guidanceIdle(data, ctx);
      s ? s.idle(idle) : domRender(`🙌 ${idle.title}<br/><small style="font-size:18px;opacity:.6">${idle.sub}</small>`);
      setHud(idle.title, idle.sub);
      break;
    }
    case "guidance_end": {
      const isV3Block = data?.block != null || data?.cond != null;
      const nextSub = isV3Block
        ? (data?.inter_round ? "准备进入下一块" : "准备进入采集")
        : (data?.inter_round ? "准备进入下一轮" : "准备进入标定");
      s ? s.idle({ title: "引导完成", sub: nextSub }) : domRender("✔");
      setHud("引导完成", nextSub);
      break;
    }
    case "round_end": {
      const idle = roundIdle(data, mode, ctx);
      const isV3Block = data?.block != null || data?.cond != null;
      const n = isV3Block
        ? (data?.block ?? data?.round ?? ctx?.round ?? "?")
        : resolveRoundNo(ctx, data);
      const doneSub = isV3Block ? `第 ${n} 块已完成` : `第 ${n} 轮已完成`;
      s ? s.idle({ title: `${idle.title}结束`, sub: doneSub }) : domRender("✔");
      setHud(`${idle.title}结束`, doneSub);
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
      showPromptBoxes(`${restText}${restDur}`, MI_GUIDANCE[0], { cross: false });
      break;
    }
    case "prep":
      s ? s.fixation() : domRender("➕");
      showPromptBoxes("", "注视十字，保持放松", { cross: true });
      break;
    case "cue": {
      clearCueGate();
      const body = cueBodyPlain(label, data);
      const title = cueTitle(label, data);
      const sub = data?.cue_sub || miSubtext(label);
      const holdMs = Math.max(0, Number(data?.cue_s ?? 1) * 1000);
      if (s) s.cue(label);
      else domRender(`cue<br/>${body}`);
      showPromptBoxes(title, sub, { cross: false, cueSplit: true, cueBody: body });
      _cueGate = { until: performance.now() + holdMs, timer: null };
      break;
    }
    case "mi": {
      const remain = _cueGate ? _cueGate.until - performance.now() : 0;
      if (remain > 40) {
        if (_cueGate.timer != null) clearTimeout(_cueGate.timer);
        const snap = {
          label,
          mode,
          data: data && typeof data === "object" ? { ...data } : data,
        };
        _cueGate.timer = setTimeout(() => {
          clearCueGate();
          applyMiStage(S(), snap.label, snap.mode, snap.data);
        }, remain);
        break;
      }
      clearCueGate();
      applyMiStage(s, label, mode, data);
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
      { stage: "cue", ctx: { label, mode }, data: { cue_s: 1 } },
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
  const gap = (st) => ({
    inter_trial_rest: 1200, cue: 1000, mi: 1200, judge: 450, iti: 1200, trial_end: 1000 }[st] ?? 800);
  let i = 0;
  (function next() {
    if (i >= evts.length) return;
    const e = evts[i++]; handleV2Stage(e.stage, e.ctx, e.data); setTimeout(next, gap(e.stage)); })();
  return true;
}
