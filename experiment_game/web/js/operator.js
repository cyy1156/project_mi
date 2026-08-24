const WS_URL =
  new URLSearchParams(location.search).get("ws") ||
  `ws://${location.hostname || "127.0.0.1"}:8765`;

const STORAGE_KEY = "experiment_game_operator_defaults_v1";

const el = {
  wsStatus: document.getElementById("ws-status"),
  setup: document.getElementById("view-setup"),
  run: document.getElementById("view-run"),
  summary: document.getElementById("view-summary"),
  form: document.getElementById("setup-form"),
  errors: document.getElementById("form-errors"),
  acqWarn: document.getElementById("acq-warning"),
  guiHint: document.getElementById("gui-hint"),
  deviceFs: document.getElementById("device-fieldset"),
  saveHint: document.getElementById("save-hint"),
  runSummary: document.getElementById("run-summary"),
  popupWarn: document.getElementById("popup-warn"),
  phaseSteps: document.getElementById("phase-steps"),
  stPhase: document.getElementById("st-phase"),
  stStage: document.getElementById("st-stage"),
  stTrial: document.getElementById("st-trial"),
  stLabel: document.getElementById("st-label"),
  stObject: document.getElementById("st-object"),
  stScene: document.getElementById("st-scene"),
  stReject: document.getElementById("st-reject"),
  stAcq: document.getElementById("st-acq"),
  linkLine: document.getElementById("link-line"),
  summaryMsg: document.getElementById("summary-msg"),
  summaryRoot: document.getElementById("summary-root"),
  summaryFiles: document.getElementById("summary-files"),
  verifyBadge: document.getElementById("verify-badge"),
  phase4Msg: document.getElementById("phase4-msg"),
  reuseBar: document.getElementById("reuse-bar"),
  reuseSummary: document.getElementById("reuse-summary"),
  portsHint: document.getElementById("ports-hint"),
  portList: document.getElementById("serial-port-list"),
  portInput: document.getElementById("serial_port_input"),
  setupTimeline: document.getElementById("setup-timeline"),
  timingHint: document.getElementById("timing-hint"),
  runTimeline: document.getElementById("run-timeline"),
  runTimingHint: document.getElementById("run-timing-hint"),
  v2Panel: document.getElementById("v2-panel"),
  v2StageHint: document.getElementById("v2-stage-hint"),
  v2GateAcc: document.getElementById("v2-gate-acc"),
  v2GateStatus: document.getElementById("v2-gate-status"),
  v2GateN: document.getElementById("v2-gate-n"),
  v2GateCurve: document.getElementById("v2-gate-curve"),
  btnV2Guidance: document.getElementById("btn-v2-guidance"),
  setupTimelineV2: document.getElementById("setup-timeline-v2"),
  timingHintV2: document.getElementById("timing-hint-v2"),
  phaseStepsV2: document.getElementById("phase-steps-v2"),
  stV2Round: document.getElementById("st-v2-round"),
  stV2Score: document.getElementById("st-v2-score"),
  v2CalProg: document.getElementById("v2-cal-prog"),
  v2GameProg: document.getElementById("v2-game-prog"),
  v2Subblock: document.getElementById("v2-subblock"),
  v2FtStatus: document.getElementById("v2-ft-status"),
  v2ScoreNum: document.getElementById("v2-score-num"),
  v2ScoreFill: document.getElementById("v2-score-fill"),
  v2WeakMi: document.getElementById("v2-weak-mi"),
  v2AcceptDetail: document.getElementById("v2-accept-detail"),
  p4SummaryV2: document.getElementById("p4-summary-v2"),
  runAlert: document.getElementById("run-alert"),
  v2GuidanceCountdown: document.getElementById("v2-guidance-countdown"),
  v3Panel: document.getElementById("v3-panel"),
  v3BlockProg: document.getElementById("v3-block-prog"),
  v3TrialProg: document.getElementById("v3-trial-prog"),
  v3Cond: document.getElementById("v3-cond"),
  v3Eeg: document.getElementById("v3-eeg"),
  v3PowerBars: document.getElementById("v3-power-bars"),
  v3FeatureCards: document.getElementById("v3-feature-cards"),
  v3HatCheck: document.getElementById("v3-hat-check"),
  v3BlockAcc: document.getElementById("v3-block-acc"),
  v3BlockErd: document.getElementById("v3-block-erd"),
  v3BlockLat: document.getElementById("v3-block-lat"),
  btnV3Guidance: document.getElementById("btn-v3-guidance"),
  btnRestart: document.getElementById("btn-restart"),
  phaseStepsV3: document.getElementById("phase-steps-v3"),
  v3EegGain: document.getElementById("v3-eeg-gain"),
  v3BlockNvalid: document.getElementById("v3-block-nvalid"),
  v3BlockNlr: document.getElementById("v3-block-nlr"),
  v3SummaryDetail: document.getElementById("v3-summary-detail"),
  v3ChGain: document.getElementById("v3-ch-gain"),
  v4Panel: document.getElementById("v4-panel"),
  v4VerdictDot: document.getElementById("v4-verdict-dot"),
  v4VerdictText: document.getElementById("v4-verdict-text"),
  v4Streak: document.getElementById("v4-streak"),
  v4StreakFill: document.getElementById("v4-streak-fill"),
  v4Headline: document.getElementById("v4-headline"),
  v4Metrics: document.getElementById("v4-metrics"),
  v4Channels: document.getElementById("v4-channels"),
  v4Problems: document.getElementById("v4-problems"),
  btnV4ToV3: document.getElementById("btn-v4-to-v3"),
  phaseStepsV4: document.getElementById("phase-steps-v4"),
  v4SummaryDetail: document.getElementById("v4-summary-detail"),
};

const V3_LABEL_NAMES = { 0: "静息", 1: "左手", 2: "右手" };
const V3_MU_ERD_OK = -15;
const V3_LAT_OK = 8;
const V3_CH = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"];
const V4_CH = V3_CH;
const V4_STATE = {
  passRequired: 5,
  achievedPass: false,
  streak: 0,
  verdict: "fail",
  perChannel: null,
  liveByName: {},
  activeIdx: null,
};
const V3_PX_PER_SEC = 80;
const V3_WINDOW_SEC = 9;
const V3_LEFT_PAD = 44;
const V3_EEG_STATE = {
  times: [],
  vals: V3_CH.map(() => []),
  events: [],
  timeAnchor: null,
  gainUv: 50,
  baselineReady: false,
  baselineMu: [],
  baselineBeta: [],
  hatVerdict: null,
};
let V3_POWER_ROWS = null;
let V3_POWER_NOTE = null;

function resetV3EegState() {
  V3_EEG_STATE.times = [];
  V3_EEG_STATE.vals = V3_CH.map(() => []);
  V3_EEG_STATE.events = [];
  V3_EEG_STATE.timeAnchor = null;
  V3_EEG_STATE.gainUv = 50;
  V3_EEG_STATE.baselineReady = false;
  V3_EEG_STATE.baselineMu = [];
  V3_EEG_STATE.baselineBeta = [];
  V3_EEG_STATE.hatVerdict = null;
  V3_POWER_ROWS = null;
  V3_POWER_NOTE = null;
}

/** 链路面板固件串：去掉串口 ``v`` 响应里的二进制前缀 */
function sanitizeFirmwareDisplay(raw) {
  const s = String(raw || "")
    .replace(/[^\x20-\x7E]/g, "")
    .trim();
  if (!s) return "";
  const m =
    s.match(/Firmware:\s*v[\d.]+/i) || s.match(/OpenBCI[\w .-]{0,40}/i);
  return m ? m[0] : s.slice(0, 48);
}

/** F3：Cyton 链路面板（streaming_hz + gap + 重连） */
function updateLinkLine(msg) {
  if (!el.linkLine) return;
  const port = msg.port || "—";
  const fwText = sanitizeFirmwareDisplay(msg.firmware);
  const fw = fwText ? ` · ${fwText}` : "";
  const hz = msg.streaming_hz != null ? `${Math.round(msg.streaming_hz)}Hz` : "—Hz";
  const gap = msg.gap_samples != null ? `gap ${msg.gap_samples}` : "gap —";
  const rc = Number(msg.reconnect_ok || 0);
  const rf = Number(msg.reconnect_fail || 0);
  const rcTxt = rc > 0 ? ` · 重连+${rc}` : rf > 0 ? ` · 重连失败${rf}` : " · 重连 0";
  let tone = "stat-ok";
  if (msg.link_dead || (msg.streaming_hz != null && msg.streaming_hz < 100)) {
    tone = "stat-bad";
  } else if (
    (msg.streaming_hz != null && msg.streaming_hz < 200) ||
    rf > 0 ||
    rc > 0
  ) {
    tone = "stat-mid";
  }
  el.linkLine.className = tone;
  let text = `${port}${fw} · ${hz} · ${gap}${rcTxt}`;
  if (msg.link_dead && msg.guidance) {
    text += ` · ${msg.guidance}`;
  }
  el.linkLine.textContent = text;
  el.linkLine.title =
    "streaming_hz=推送增量/2s；drop_rate 仅评录制质量。链路看 hz+gap+重连。";
}

function v3LslNow() {
  if (!V3_EEG_STATE.timeAnchor) return 0;
  const a = V3_EEG_STATE.timeAnchor;
  return a.lslT + (Date.now() - a.wallMs) / 1000;
}

function markV3StageEvent(stage) {
  if (!["cue", "mi", "iti", "mi_start"].includes(stage)) return;
  const key = stage === "mi_start" ? "mi" : stage;
  V3_EEG_STATE.events.push({ t: v3LslNow(), stage: key });
  while (V3_EEG_STATE.events.length > 40) V3_EEG_STATE.events.shift();
}

function appendV3EegSamples(msg) {
  const fs = Number(msg.fs_disp) || 62.5;
  const nCh = msg.data?.length || 0;
  const nSamp = nCh > 0 ? (msg.data[0]?.length || 0) : 0;
  if (!nSamp) return;
  const tEnd = Number(msg.t) || v3LslNow();
  if (!V3_EEG_STATE.timeAnchor) {
    V3_EEG_STATE.timeAnchor = { wallMs: Date.now(), lslT: tEnd };
  }
  const times = V3_EEG_STATE.times;
  const vals = V3_EEG_STATE.vals;
  for (let si = 0; si < nSamp; si++) {
    times.push(tEnd - (nSamp - 1 - si) / fs);
    for (let ci = 0; ci < V3_CH.length; ci++) {
      const v = ci < nCh ? Number(msg.data[ci][si]) : NaN;
      vals[ci].push(Number.isFinite(v) ? v : 0);
    }
  }
  const tCut = tEnd - V3_WINDOW_SEC - 1;
  let cut = 0;
  while (cut < times.length && times[cut] < tCut) cut++;
  if (cut > 0) {
    times.splice(0, cut);
    for (const arr of vals) arr.splice(0, cut);
  }
  while (V3_EEG_STATE.events.length && V3_EEG_STATE.events[0].t < tCut) {
    V3_EEG_STATE.events.shift();
  }
}

function v3ComputeGain() {
  const tWin = v3LslNow() - 10;
  const times = V3_EEG_STATE.times;
  let i0 = 0;
  while (i0 < times.length && times[i0] < tWin) i0++;
  if (i0 >= times.length) return { global: 50, perCh: V3_CH.map(() => 50) };
  const p95 = (arr) => {
    if (!arr.length) return 50;
    arr.sort((a, b) => a - b);
    return Math.max(5, (arr[Math.floor(arr.length * 0.95)] || 5) * 1.2);
  };
  const all = [];
  const perCh = V3_CH.map(() => []);
  for (let ci = 0; ci < V3_CH.length; ci++) {
    const arr = V3_EEG_STATE.vals[ci];
    for (let i = i0; i < times.length; i++) {
      const a = Math.abs(arr[i]);
      perCh[ci].push(a);
      all.push(a);
    }
  }
  return { global: p95(all), perCh: perCh.map(p95) };
}

function erdPct(p, p0) {
  if (!p0 || p0 <= 0) return null;
  return (100 * (p - p0)) / p0;
}

function erdClass(v) {
  if (v == null || !Number.isFinite(v)) return "erd-na";
  if (v <= V3_MU_ERD_OK) return "erd-ok";
  if (v <= 0) return "erd-mid";
  return "erd-bad";
}

function drawV3EegCanvas() {
  if (!el.v3Eeg) return;
  const ctx = el.v3Eeg.getContext("2d");
  if (!ctx) return;
  const W = el.v3Eeg.width;
  const H = el.v3Eeg.height;
  const tNow = v3LslNow();
  const t0 = tNow - V3_WINDOW_SEC;
  const plotW = W - V3_LEFT_PAD - 8;
  const gain = v3ComputeGain();
  V3_EEG_STATE.gainUv = gain.global;
  const perChGain = el.v3ChGain?.checked === true;
  if (el.v3EegGain) {
    el.v3EegGain.textContent = perChGain ? "逐通道增益" : `±${Math.round(gain.global)} µV`;
  }

  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);
  const nCh = V3_CH.length;
  const rowH = H / nCh;

  for (let sec = 0; sec <= V3_WINDOW_SEC; sec++) {
    const x = V3_LEFT_PAD + ((t0 + sec - t0) / V3_WINDOW_SEC) * plotW;
    ctx.strokeStyle = sec % 2 === 0 ? "#30363d" : "#21262d";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
    if (sec % 2 === 0) {
      ctx.fillStyle = "#6e7681";
      ctx.font = "9px sans-serif";
      ctx.fillText(`${sec}s`, x + 2, H - 2);
    }
  }

  for (const ev of V3_EEG_STATE.events) {
    if (ev.t < t0 || ev.t > tNow) continue;
    const x = V3_LEFT_PAD + ((ev.t - t0) / V3_WINDOW_SEC) * plotW;
    ctx.strokeStyle = ev.stage === "cue" ? "#d29922" : ev.stage === "mi" ? "#3fb950" : "#8b949e";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  const times = V3_EEG_STATE.times;
  for (let ci = 0; ci < nCh; ci++) {
    const yMid = ci * rowH + rowH / 2;
    const gUv = perChGain ? gain.perCh[ci] : gain.global;
    const scale = (rowH * 0.4) / gUv;
    ctx.fillStyle = "#8b949e";
    ctx.font = "10px sans-serif";
    ctx.fillText(V3_CH[ci], 4, yMid + 4);
    // µV 刻度条放在通道名右侧，避免与文字重叠
    const barH = rowH * 0.35;
    ctx.strokeStyle = "#484f58";
    ctx.strokeRect(26, yMid - barH / 2, 8, barH);
    ctx.fillStyle = "#484f58";
    ctx.fillRect(26, yMid - 1, 8, 2);
    if (perChGain) {
      ctx.fillStyle = "#6e7681";
      ctx.fillText(`±${Math.round(gUv)}`, V3_LEFT_PAD + plotW - 34, yMid + 4);
    }

    const arr = V3_EEG_STATE.vals[ci];
    if (times.length < 2) continue;
    const nPx = Math.max(1, Math.floor(plotW));
    ctx.strokeStyle = "#58a6ff";
    ctx.beginPath();
    let started = false;
    // 双指针按像素分桶：times 升序，单通道总复杂度 O(nSamples + nPx)
    let ri = 0;
    while (ri < times.length && times[ri] < t0) ri++;
    for (let px = 0; px < nPx; px++) {
      const ta = t0 + (px / nPx) * V3_WINDOW_SEC;
      const tb = t0 + ((px + 1) / nPx) * V3_WINDOW_SEC;
      while (ri < times.length && times[ri] < ta) ri++;
      let vmin = Infinity;
      let vmax = -Infinity;
      for (let j = ri; j < times.length && times[j] < tb; j++) {
        const v = arr[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
      if (!Number.isFinite(vmin)) continue;
      const x = V3_LEFT_PAD + px;
      const y1 = yMid - vmax * scale;
      const y2 = yMid - vmin * scale;
      if (!started) {
        ctx.moveTo(x, y1);
        started = true;
      } else {
        ctx.lineTo(x, y1);
      }
      ctx.lineTo(x, y2);
    }
    if (started) ctx.stroke();
  }
}

function v3SignalHint(reason, msg = {}) {
  if (reason === "artifact") return "大幅伪迹：减少移动、检查电极接触";
  if (reason === "dead_channel") {
    const idx = msg.dead_channel_idx ?? msg.signal_metrics?.dead_channel_idx;
    const ch = idx != null && V3_CH[idx] ? V3_CH[idx] : "?";
    return `通道 ${ch} 未接好：补胶/按紧`;
  }
  if (reason === "common_mode") return "参考电极接触不良：检查 SRB2/Bias";
  return "";
}

function v3SignalNoteText(msg) {
  const reason = msg.signal_reason;
  const hint = v3SignalHint(reason, msg);
  const base = `当前信号质量不足${reason ? `（${reason}）` : ""}`;
  const tail = hint ? `：${hint}` : "";
  const baselineNote =
    V3_EEG_STATE.hatVerdict && V3_EEG_STATE.hatVerdict !== "pass"
      ? "，基线可能无效"
      : "";
  return `${base}${tail}，功率条仅供参考${baselineNote}`;
}

function renderV3PowerBars(msg) {
  if (!el.v3PowerBars) return;
  const sigOk = msg.signal_ok !== false;
  if (!V3_EEG_STATE.baselineReady) {
    V3_POWER_ROWS = null;
    V3_POWER_NOTE = null;
    const note = sigOk ? "" : `<div class="v3-pbar-signal">${v3SignalNoteText(msg)}，基线可能无效</div>`;
    el.v3PowerBars.innerHTML = note + `<div class="v3-pbar-wait">基线采集中…（前 60s 静息阶段结束后显示 ERD%）</div>`;
    return;
  }
  if (!V3_POWER_ROWS) {
    el.v3PowerBars.innerHTML = `<div class="v3-pbar-signal hidden"></div>` + V3_CH.map((ch) => `<div class="v3-pbar-row">
      <span class="v3-pbar-ch">${ch}</span>
      <span class="v3-pbar-track mu"><span class="fill"></span><span class="tick-15"></span></span>
      <span class="v3-pbar-num">—</span>
      <span class="v3-pbar-track beta"><span class="fill"></span></span>
      <span class="v3-pbar-num">—</span>
    </div>`).join("");
    V3_POWER_NOTE = el.v3PowerBars.querySelector(".v3-pbar-signal");
    V3_POWER_ROWS = Array.from(el.v3PowerBars.querySelectorAll(".v3-pbar-row")).map((row) => {
      const tracks = row.querySelectorAll(".v3-pbar-track");
      const nums = row.querySelectorAll(".v3-pbar-num");
      return {
        row,
        muTrack: tracks[0],
        muFill: tracks[0].querySelector(".fill"),
        muNum: nums[0],
        betaTrack: tracks[1],
        betaFill: tracks[1].querySelector(".fill"),
        betaNum: nums[1],
      };
    });
  }
  if (V3_POWER_NOTE) {
    V3_POWER_NOTE.classList.toggle("hidden", sigOk);
    if (!sigOk) V3_POWER_NOTE.textContent = v3SignalNoteText(msg);
  }
  V3_POWER_ROWS.forEach((r, i) => {
    const pMu = msg.power_mu?.[i];
    const pBeta = msg.power_beta?.[i];
    const erdMu = erdPct(pMu, V3_EEG_STATE.baselineMu[i]);
    const erdBeta = erdPct(pBeta, V3_EEG_STATE.baselineBeta[i]);
    const fmt = (v) => (v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(0)}%`);
    r.muFill.style.width = `${erdMu == null ? 0 : Math.min(100, Math.max(0, 50 - erdMu))}%`;
    r.betaFill.style.width = `${erdBeta == null ? 0 : Math.min(100, Math.max(0, 50 - erdBeta))}%`;
    // 信号质量不足时整行灰化、数值不上红绿，避免"平线全绿"误读
    const muCls = sigOk ? erdClass(erdMu) : "";
    r.muTrack.className = `v3-pbar-track mu ${muCls}`.trim();
    r.muNum.className = `v3-pbar-num ${muCls}`.trim();
    r.muNum.textContent = fmt(erdMu);
    // beta 条无独立合格线，保持中性紫色，不参与红绿判定
    r.betaTrack.className = "v3-pbar-track beta";
    r.betaNum.className = "v3-pbar-num";
    r.betaNum.textContent = fmt(erdBeta);
    r.row.classList.toggle("v3-pbar-dim", !sigOk);
    r.row.title = `mu ${pMu?.toExponential?.(2) ?? "—"} µV² · beta ${pBeta?.toExponential?.(2) ?? "—"} µV²`;
  });
}

function drawV3EegFrame(msg) {
  appendV3EegSamples(msg);
  drawV3EegCanvas();
  renderV3PowerBars(msg);
}

function v3GradeTone(grade) {
  if (grade === "明显" || grade === "基线") return "verdict-good";
  if (grade === "中等") return "verdict-mid";
  if (grade === "预热") return "verdict-warmup";
  return "verdict-weak";
}

function renderV3Checks(checks) {
  const names = {
    mu_erd_contra: "对侧 mu ERD",
    laterality: "偏侧化",
    mu_vs_betal: "mu 优于 beta",
    rest_mu_frac: "静息 mu 占比",
    time_pattern: "时间形态",
  };
  const core = ["mu_erd_contra", "laterality", "mu_vs_betal", "rest_mu_frac", "time_pattern"];
  return core
    .filter((k) => k in (checks || {}))
    .map((k) => `<span class="${checks[k] ? "chk-ok" : "chk-fail"}">${checks[k] ? "✓" : "✗"} ${names[k] || k}</span>`)
    .join(" ");
}

function appendV3FeatureCard(msg) {
  if (!el.v3FeatureCards) return;
  const f = msg.features || {};
  const tg = msg.trial_grade || msg.grade || {};
  const bg = msg.block_grade || {};
  const pj = msg.primary_judge;
  const pThree = msg.p_three || pj?.p_three;
  const label = msg.label;
  const isRest = f.is_rest || label === 0;

  let verdict = f.verdict_text || "—";
  let verdictCls = v3GradeTone(tg.grade);

  const card = document.createElement("div");
  card.className = "v3-fcard";

  if (msg.signal_bad || f.signal_bad) {
    card.innerHTML = [
      `<div class="v3-verdict verdict-weak">${f.verdict_text || "信号质量不足，本试次不计统计"}</div>`,
      `<div class="v3-fcard-head">试次 ${msg.trial_id} · ${V3_LABEL_NAMES[label] || label} · ${msg.cond || ""} · 不计统计</div>`,
    ].join("");
    el.v3FeatureCards.prepend(card);
    while (el.v3FeatureCards.children.length > 6) {
      el.v3FeatureCards.removeChild(el.v3FeatureCards.lastChild);
    }
    return;
  }

  if (isRest) {
    card.innerHTML = [
      `<div class="v3-verdict ${verdictCls}">${verdict}</div>`,
      `<div class="v3-fcard-head">试次 ${msg.trial_id} · ${V3_LABEL_NAMES[label] || label} · ${msg.cond || ""}</div>`,
      f.warmup ? `<div class="v3-rest-note">预热试次，不计 ERD</div>` : `<div class="v3-rest-note">块内 Rest 窗：${(f.n_rest_windows_before || 0) + (f.n_windows || 0)}</div>`,
    ].join("");
  } else {
    const pred = pj ? Number(pj.pred) : null;
    const predName = pred != null ? V3_LABEL_NAMES[pred] || pred : "—";
    const correct = pred != null && pred === label;
    const contraCh = f.contra_ch || (label === 1 ? "C4" : "C3");
    const muErd = f.mu_erd || {};
    const chCells = ["C3", "C4", "CP3", "CP4"]
      .map((ch) => {
        const v = muErd[ch];
        const cls = erdClass(v);
        const bold = ch === contraCh ? " contra" : "";
        return `<span class="v3-erd-cell ${cls}${bold}">${ch} ${v != null ? v.toFixed(0) : "—"}%</span>`;
      })
      .join("");
    const lat = f.laterality_pp;
    const latCls = lat != null && lat >= V3_LAT_OK ? "erd-ok" : "erd-mid";
    const pBars = Array.isArray(pThree)
      ? ["Rest", "Left", "Right"]
          .map((name, i) => {
            const p = pThree[i] ?? 0;
            return `<div class="v3-pbar3"><span>${name}</span><span class="track"><span style="width:${Math.round(p * 100)}%"></span></span><span>${(p * 100).toFixed(0)}%</span></div>`;
          })
          .join("")
      : "";

    card.innerHTML = [
      `<div class="v3-verdict ${verdictCls}">${verdict}</div>`,
      `<div class="v3-fcard-head">试次 ${msg.trial_id} · ${V3_LABEL_NAMES[label] || label} · ${msg.cond || ""}${msg.valid === false ? " · 无效" : ""}</div>`,
      `<div class="v3-trial-row">模型：<strong>${predName}</strong> ${pred != null ? (correct ? '<span class="chk-ok">✓ 正确</span>' : '<span class="chk-fail">✗ 错误</span>') : ""}</div>`,
      pBars ? `<div class="v3-pthree">${pBars}</div>` : "",
      `<div class="v3-erd-grid-label">各通道 mu ERD%（vs 块内 Rest 基线）</div>`,
      `<div class="v3-erd-grid">${chCells}</div>`,
      `<div class="v3-lat-row">偏侧 ${lat != null ? lat.toFixed(1) : "—"}pp <span class="v3-thr">（≥${V3_LAT_OK}pp）</span> <span class="${latCls}">${lat != null && lat >= V3_LAT_OK ? "✓" : ""}</span></div>`,
      `<div class="v3-block-grade"><span class="v3-block-label">块累计</span> ${bg.grade || "—"} · n=${f.block_n_mi_trials ?? "—"} MI / ${f.block_n_rest_trials ?? "—"} Rest</div>`,
      `<div class="v3-checks">${renderV3Checks(bg.checks)}</div>`,
    ].join("");
  }

  el.v3FeatureCards.prepend(card);
  while (el.v3FeatureCards.children.length > 6) {
    el.v3FeatureCards.removeChild(el.v3FeatureCards.lastChild);
  }
}

function updateV3Block(msg) {
  if (el.v3BlockProg) el.v3BlockProg.textContent = `${msg.block_idx ?? "—"}/${msg.blocks_total ?? 2}`;
  if (el.v3TrialProg) el.v3TrialProg.textContent = `${msg.trial_done ?? 0}/${msg.trials_per_block ?? "—"}`;
  if (el.v3Cond) el.v3Cond.textContent = msg.cond || "—";
  if (el.v3BlockNvalid) {
    let txt = String(msg.n_valid ?? "—");
    if (msg.n_signal_bad) txt += ` · 信号剔除 ${msg.n_signal_bad}`;
    el.v3BlockNvalid.textContent = txt;
  }
  if (el.v3BlockNlr) el.v3BlockNlr.textContent = String(msg.n_lr ?? "—");
  if (msg.acc_argmax != null && el.v3BlockAcc) {
    const cls = msg.acc_argmax >= 0.6 ? "stat-ok" : msg.acc_argmax >= 0.45 ? "stat-mid" : "stat-bad";
    el.v3BlockAcc.innerHTML = `<span class="${cls}">${(msg.acc_argmax * 100).toFixed(1)}%</span>`;
  }
  if (msg.mu_erd_contra_mean != null && el.v3BlockErd) {
    const cls = msg.mu_erd_contra_mean <= V3_MU_ERD_OK ? "stat-ok" : "stat-mid";
    el.v3BlockErd.innerHTML = `<span class="${cls}">${msg.mu_erd_contra_mean}%</span>`;
  }
  if (msg.laterality_pp_mean != null && el.v3BlockLat) {
    const cls = msg.laterality_pp_mean >= V3_LAT_OK ? "stat-ok" : "stat-mid";
    el.v3BlockLat.innerHTML = `<span class="${cls}">${msg.laterality_pp_mean}pp</span>`;
  }
}

function showV3HatCheck(msg) {
  if (!el.v3HatCheck) return;
  const hat = msg.hat_check || {};
  const verdict = msg.hat_verdict || hat.verdict;
  const text = msg.hat_message || hat.message;
  if (!verdict || !text) {
    el.v3HatCheck.classList.add("hidden");
    el.v3HatCheck.textContent = "";
    V3_EEG_STATE.hatVerdict = null;
    return;
  }
  V3_EEG_STATE.hatVerdict = verdict;
  el.v3HatCheck.textContent = text;
  el.v3HatCheck.className = `v3-hat-check hat-${verdict}`;
  el.v3HatCheck.classList.remove("hidden");
}

function handleV3Baseline(msg) {
  V3_EEG_STATE.baselineReady = true;
  V3_EEG_STATE.baselineMu = msg.baseline_mu || [];
  V3_EEG_STATE.baselineBeta = msg.baseline_beta || [];
  showV3HatCheck(msg);
}

const V3_OVERRIDE_KEYS = [
  ["v3_blocks", "blocks", "int"],
  ["v3_trials_per_block", "trials_per_block", "int"],
  ["v3_baseline_rest_s", "baseline_rest_s", "float"],
  ["v3_block_gap_s", "block_gap_s", "float"],
  ["v3_prep_s", "prep_s", "float"],
  ["v3_cue_s", "cue_s", "float"],
  ["v3_imagine_s", "imagine_s", "float"],
  ["v3_iti_s", "iti_s", "float"],
];

let guidanceTimer = null;

function playAlertBeep(kind = "alert") {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sine";
    osc.frequency.value = kind === "guidance" ? 880 : 440;
    gain.gain.value = 0.08;
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + (kind === "guidance" ? 0.25 : 0.45));
    osc.stop(ctx.currentTime + (kind === "guidance" ? 0.25 : 0.45));
  } catch {
    /* WebAudio unavailable */
  }
}

function showRunAlert(text, kind = "degraded") {
  if (!el.runAlert) return;
  if (!text) {
    el.runAlert.classList.add("hidden");
    el.runAlert.textContent = "";
    el.runAlert.className = "run-alert hidden";
    return;
  }
  el.runAlert.textContent = text;
  el.runAlert.className = `run-alert ${kind}`;
  el.runAlert.classList.remove("hidden");
}

function stopGuidanceCountdown() {
  if (guidanceTimer) {
    clearInterval(guidanceTimer);
    guidanceTimer = null;
  }
  el.btnV2Guidance?.classList.remove("guidance-pulse");
  el.v2GuidanceCountdown?.classList.add("hidden");
  el.v2GuidanceCountdown?.classList.remove("urgent");
}

function startGuidanceCountdown(totalSec) {
  stopGuidanceCountdown();
  if (!totalSec || !el.v2GuidanceCountdown) return;
  let left = Math.ceil(Number(totalSec));
  const tick = () => {
    if (el.v2GuidanceCountdown) {
      el.v2GuidanceCountdown.textContent = `引导倒计时 ${left}s — 请完成抬臂引导后点「确认动觉引导完成」`;
      el.v2GuidanceCountdown.classList.toggle("urgent", left <= 15);
      el.v2GuidanceCountdown.classList.remove("hidden");
    }
    if (left <= 0) {
      stopGuidanceCountdown();
      return;
    }
    left -= 1;
  };
  el.btnV2Guidance?.classList.add("guidance-pulse");
  playAlertBeep("guidance");
  tick();
  guidanceTimer = setInterval(tick, 1000);
}

const V2_TIMING_PRESETS = {
  standard: { prep_s: 2, cue_s: 2, imagine_s: 6, iti_s: 3 },
  debug4: { prep_s: 2, cue_s: 2, imagine_s: 4, iti_s: 2 },
};

const V2_OVERRIDE_KEYS = [
  ["v2_cal_rounds_min", "cal_rounds_min", "int"],
  ["v2_cal_rounds_max", "cal_rounds_max", "int"],
  ["v2_game_rounds", "game_rounds", "int"],
  ["v2_trials_per_round", "trials_per_round", "int"],
  ["v2_ft_trials_per_round", "ft_trials_per_round", "int"],
  ["v2_quiz_trials_per_round", "quiz_trials_per_round", "int"],
  ["v2_game_trials_per_round", "game_trials_per_round", "int"],
  ["v2_prep_s", "prep_s", "float"],
  ["v2_cue_s", "cue_s", "float"],
  ["v2_imagine_s", "imagine_s", "float"],
  ["v2_iti_s", "iti_s", "float"],
  ["v2_gate_enter_three", "gate_enter_three", "float"],
  ["v2_gate_min_quiz_trials", "gate_min_quiz_trials", "int"],
  ["v2_judgment_step_s", "judgment_step_s", "float"],
  ["v2_judgment_half_weight_until_s", "judgment_half_weight_until_s", "float"],
  ["v2_score_early_stop", "score_early_stop", "float"],
  ["v2_score_invalid_max", "score_invalid_max", "float"],
  ["v2_wrong_class_abort", "wrong_class_abort", "float"],
  ["v2_consecutive_invalid_abort", "consecutive_invalid_abort", "int"],
  ["v2_ft_min_valid_trials", "ft_min_valid_trials", "int"],
  ["v2_group_lr", "group_lr", "float"],
  ["v2_replay_ratio", "replay_ratio", "float"],
  ["v2_drift_patience", "drift_patience", "int"],
  ["v2_task_p_on", "task_p_on", "float"],
  ["v2_ft_epochs", "ft_epochs", "int"],
  ["v2_ft_batch_size", "ft_batch_size", "int"],
];

function readV2TimingFromForm() {
  return {
    fixation_s: Number(el.form.elements.namedItem("v2_prep_s")?.value) || 0,
    cue_s: Number(el.form.elements.namedItem("v2_cue_s")?.value) || 0,
    mi_s: Number(el.form.elements.namedItem("v2_imagine_s")?.value) || 0,
    post_mi_hold_s: 0,
    rest_s: 0,
    transition_s: Number(el.form.elements.namedItem("v2_iti_s")?.value) || 0,
  };
}

function readV3Overrides() {
  const o = {};
  for (const [formName, key, typ] of V3_OVERRIDE_KEYS) {
    const node = el.form.elements.namedItem(formName);
    if (!node || node.value === "") continue;
    const n = Number(node.value);
    if (!Number.isFinite(n)) continue;
    o[key] = typ === "int" ? Math.round(n) : n;
  }
  return o;
}

function readV2Overrides() {
  const o = {};
  for (const [formName, key, typ] of V2_OVERRIDE_KEYS) {
    const node = el.form.elements.namedItem(formName);
    if (!node || node.value === "") continue;
    const n = Number(node.value);
    if (!Number.isFinite(n)) continue;
    o[key] = typ === "int" ? Math.round(n) : n;
  }
  if (o.game_rounds != null) {
    o.game_rounds_min = Math.min(o.game_rounds_min ?? 1, o.game_rounds);
    o.game_rounds_max = Math.max(o.game_rounds_max ?? 3, o.game_rounds);
  }
  if (o.cal_rounds_min != null && o.cal_rounds_max != null && o.cal_rounds_min > o.cal_rounds_max) {
    const t = o.cal_rounds_min;
    o.cal_rounds_min = o.cal_rounds_max;
    o.cal_rounds_max = t;
  }
  return o;
}

function applyV2OverridesToForm(ov) {
  if (!ov) return;
  for (const [formName, key] of V2_OVERRIDE_KEYS) {
    if (ov[key] == null) continue;
    const node = el.form.elements.namedItem(formName);
    if (node) node.value = ov[key];
  }
  renderSetupTimelineV2();
}

function updateV2TrialComposeHint() {
  const hint = document.getElementById("v2-trial-compose-hint");
  if (!hint) return;
  const total = Number(el.form.elements.namedItem("v2_trials_per_round")?.value) || 0;
  const ft = Number(el.form.elements.namedItem("v2_ft_trials_per_round")?.value) || 0;
  const quiz = Number(el.form.elements.namedItem("v2_quiz_trials_per_round")?.value) || 0;
  const gameN = Number(el.form.elements.namedItem("v2_game_trials_per_round")?.value) || 0;
  const ok = ft + quiz === total && total > 0;
  hint.textContent = ok
    ? `标定每轮 ${total} 试次 = 前 ${ft} 训练(FT) + 后 ${quiz} 小考(测试，不进训练)；游戏每轮 ${gameN} 试次（组级 FT）。`
    : `⚠ 构成不合法：FT(${ft}) + 小考(${quiz}) 须等于每轮总试次(${total})。`;
  hint.style.color = ok ? "" : "var(--danger)";
}

function renderSetupTimelineV2() {
  const t = readV2TimingFromForm();
  const { total } = renderTimeline(el.setupTimelineV2, t) || { total: 0 };
  updateV2TrialComposeHint();
  if (el.timingHintV2) {
    const calMin = Number(el.form.elements.namedItem("v2_cal_rounds_min")?.value) || 4;
    const calMax = Number(el.form.elements.namedItem("v2_cal_rounds_max")?.value) || 6;
    const gameR = Number(el.form.elements.namedItem("v2_game_rounds")?.value) || 2;
    const perCal = Number(el.form.elements.namedItem("v2_trials_per_round")?.value) || 18;
    const perGame = Number(el.form.elements.namedItem("v2_game_trials_per_round")?.value) || 16;
    const trials = calMax * perCal + gameR * perGame;
    const est = trials * total;
    el.timingHintV2.textContent =
      `单 trial = ${total}s（建议 ≤13）· 标定 ${calMin}–${calMax} 轮×${perCal} + 游戏 ${gameR}×${perGame}` +
      ` · 纯试次 ≈ ${Math.round(est / 60)} 分钟（任务书口径约 60–75 min 含间隙）`;
  }
}

function syncProtocolLockUi() {
  const locked = el.form.querySelector('[name="protocol_locked"]')?.checked !== false;
  document.querySelectorAll(".v2-lockable, .v3-lockable").forEach((fs) => {
    fs.querySelectorAll("input, select, button").forEach((node) => {
      if (node.name === "protocol_locked") return;
      node.disabled = locked;
    });
    fs.querySelectorAll(".lock-badge").forEach((b) => b.classList.toggle("hidden", !locked));
  });
}

function isV2Mode() {
  return (el.form.querySelector('input[name="phase_mode"]:checked') || {}).value === "v2_session";
}

function isV3Mode() {
  return (el.form.querySelector('input[name="phase_mode"]:checked') || {}).value === "v3_session";
}

function isV4Mode() {
  return (el.form.querySelector('input[name="phase_mode"]:checked') || {}).value === "v4_session";
}

function resetV4State() {
  V4_STATE.passRequired = 5;
  V4_STATE.achievedPass = false;
  V4_STATE.streak = 0;
  V4_STATE.verdict = "fail";
  V4_STATE.perChannel = null;
  V4_STATE.liveByName = {};
  V4_STATE.activeIdx = null;
}

function v4VerdictClass(v) {
  if (v === "pass") return "pass";
  if (v === "warn") return "warn";
  return "fail";
}

function renderV4Metrics(msg) {
  if (!el.v4Metrics) return;
  const m = msg.metrics || {};
  const items = [
    {
      k: "中位 std (µV)",
      v: m.median_std_uv != null ? m.median_std_uv.toFixed(1) : "—",
      ok: m.median_std_ok,
    },
    {
      k: "峰峰值 (µV)",
      v: m.ptp_uv != null ? m.ptp_uv.toFixed(1) : "—",
      ok: m.ptp_ok,
    },
    {
      k: "有效通道",
      v: m.active_channels != null ? `${m.active_channels}/8` : "—",
      ok: m.ch_ok,
    },
    {
      k: "共模比",
      v: m.common_mode_ratio != null ? `${Math.round(m.common_mode_ratio * 100)}%` : "—",
      ok: m.cm_ok,
    },
  ];
  el.v4Metrics.innerHTML = items
    .map(
      (it) =>
        `<div class="v4-metric ${it.ok === true ? "ok" : it.ok === false ? "bad" : ""}"><span class="k">${it.k}</span><span class="v">${it.v}</span></div>`
    )
    .join("");
}

function renderV4ChannelMap(perChannel, liveByName, activeIdx) {
  if (!el.v4Channels) return;
  const qc = perChannel && perChannel.length ? perChannel : V4_CH.map((name, idx) => ({ idx, name, ok: null, reason: null }));
  el.v4Channels.innerHTML = qc
    .map((ch) => {
      const live = liveByName[ch.name] || {};
      const stdTxt = live.std_uv != null ? `${live.std_uv}µV` : "—";
      let cls = "idle";
      if (ch.ok === true) cls = "ok";
      else if (ch.ok === false) cls = "bad";
      if (activeIdx != null && ch.idx === activeIdx && (live.std_uv || 0) >= 8) {
        cls = ch.ok === true ? "ok" : "reacting";
      }
      const reason = ch.reason ? `<span class="v4-ch-reason">${ch.reason}</span>` : "";
      return `<div class="v4-ch-cell ${cls}"><strong>${ch.name}</strong><span class="v4-ch-std">${stdTxt}</span>${reason}</div>`;
    })
    .join("");
}

function onV4Live(msg) {
  const live = msg.per_channel_live || [];
  V4_STATE.liveByName = {};
  for (const ch of live) {
    V4_STATE.liveByName[ch.name] = ch;
  }
  V4_STATE.activeIdx = msg.active_idx;
  renderV4ChannelMap(V4_STATE.perChannel, V4_STATE.liveByName, V4_STATE.activeIdx);
}

function renderV4Diagnosis(problems) {
  if (!el.v4Problems) return;
  const list = problems || [];
  if (!list.length) {
    el.v4Problems.innerHTML = "<li>暂无问题</li>";
    return;
  }
  el.v4Problems.innerHTML = list
    .map((p) => `<li>${p.hint || p.detail || p.reason}${p.channel ? ` (${p.channel})` : ""}</li>`)
    .join("");
}

function updateV4Quality(msg) {
  const verdict = msg.rolling_verdict || (msg.window_ok ? "warn" : "fail");
  V4_STATE.verdict = verdict;
  V4_STATE.streak = Number(msg.pass_streak || 0);
  if (el.v4VerdictDot) {
    el.v4VerdictDot.className = `v4-dot ${v4VerdictClass(verdict)}`;
  }
  if (el.v4VerdictText) {
    const labels = { pass: "PASS · 信号稳定", warn: "WARN · 接近达标", fail: "FAIL · 需修电极" };
    el.v4VerdictText.textContent = labels[verdict] || "检测中";
  }
  const req = V4_STATE.passRequired;
  if (el.v4Streak) el.v4Streak.textContent = `${V4_STATE.streak}/${req}`;
  if (el.v4StreakFill) {
    el.v4StreakFill.style.width = `${Math.min(100, (100 * V4_STATE.streak) / Math.max(1, req))}%`;
  }
  const top = (msg.problems || [])[0];
  if (el.v4Headline) {
    el.v4Headline.textContent = top?.hint || (msg.window_ok ? "本窗达标，保持静坐…" : "请按诊断提示检查电极");
  }
  renderV4Metrics(msg);
  V4_STATE.perChannel = msg.per_channel || null;
  renderV4ChannelMap(V4_STATE.perChannel, V4_STATE.liveByName, V4_STATE.activeIdx);
  renderV4Diagnosis(msg.problems);
  if (el.btnV4ToV3) el.btnV4ToV3.disabled = !V4_STATE.achievedPass;
}

function onV4Start(msg) {
  resetV4State();
  V4_STATE.passRequired = Number(
    msg.pass_streak_required || msg.v4_config_effective?.pass_streak_required || 5
  );
  if (el.v4Channels) renderV4ChannelMap([]);
  if (el.v4Metrics) el.v4Metrics.innerHTML = "";
  if (el.v4Problems) el.v4Problems.innerHTML = "";
  if (el.v4Headline) el.v4Headline.textContent = "请静坐放松，正在评估信号…";
  if (el.btnV4ToV3) el.btnV4ToV3.disabled = true;
  setPhaseStepV4("check");
}

function onV4Pass(msg) {
  V4_STATE.achievedPass = true;
  if (el.v4Headline) el.v4Headline.textContent = msg.message || "信号稳定，可以开始 v3";
  if (el.btnV4ToV3) el.btnV4ToV3.disabled = false;
  setPhaseStepV4("stable");
}

function onV4Summary(msg) {
  V4_STATE.verdict = msg.verdict || "fail";
  if (el.v4VerdictDot) el.v4VerdictDot.className = `v4-dot ${v4VerdictClass(V4_STATE.verdict)}`;
  if (el.v4VerdictText) {
    el.v4VerdictText.textContent =
      msg.verdict === "pass" ? "PASS" : msg.verdict === "warn" ? "WARN" : "FAIL";
  }
  if (el.v4Headline) el.v4Headline.textContent = msg.recommendation || "检测结束";
  setPhaseStepV4("done");
}

function setPhaseStepV4(step) {
  const order = ["check", "stable", "done"];
  const idx = order.indexOf(step);
  el.phaseStepsV4?.querySelectorAll("li").forEach((li) => {
    const p = li.getAttribute("data-phase");
    const i = order.indexOf(p);
    li.classList.toggle("active", p === step);
    li.classList.toggle("done", idx >= 0 && i >= 0 && i < idx);
  });
}

function setV4RunPanel(on) {
  if (on) {
    setV2RunPanel(false);
    setV3RunPanel(false);
    resetV4State();
  }
  el.v4Panel?.classList.toggle("hidden", !on);
  el.phaseSteps?.classList.toggle("hidden", on || isV2Mode() || isV3Mode());
  el.phaseStepsV2?.classList.toggle("hidden", true);
  el.phaseStepsV3?.classList.toggle("hidden", true);
  el.phaseStepsV4?.classList.toggle("hidden", !on);
  document.querySelector(".tl-block")?.classList.toggle("hidden", on);
  document.getElementById("st-object-wrap")?.classList.toggle("hidden", on || isV2Mode());
  document.getElementById("st-scene-wrap")?.classList.toggle("hidden", on || isV2Mode());
  document.getElementById("btn-gate")?.classList.toggle("hidden", on || isV2Mode());
  document.getElementById("btn-split")?.classList.toggle("hidden", on || isV2Mode());
}

function syncPhaseModeUi() {
  const v2 = isV2Mode();
  const v3 = isV3Mode();
  const v4 = isV4Mode();
  document.querySelectorAll(".phase2-only").forEach((node) => {
    node.classList.toggle("hidden", v2 || v3 || v4);
  });
  document.querySelectorAll(".v2-only").forEach((node) => {
    node.classList.toggle("hidden", !v2);
  });
  document.querySelectorAll(".v3-only").forEach((node) => {
    node.classList.toggle("hidden", !v3);
  });
  document.querySelectorAll(".v4-only").forEach((node) => {
    node.classList.toggle("hidden", !v4);
  });
  syncProtocolLockUi();
  if (v2) renderSetupTimelineV2();
  else if (!v3 && !v4) renderSetupTimeline();
}

function setV3RunPanel(on) {
  if (on) {
    setV2RunPanel(false);
    setV4RunPanel(false);
    resetV3EegState();
    if (el.v3FeatureCards) el.v3FeatureCards.innerHTML = "";
    if (el.v3HatCheck) {
      el.v3HatCheck.textContent = "";
      el.v3HatCheck.className = "v3-hat-check hidden";
    }
    if (el.v3PowerBars) {
      el.v3PowerBars.innerHTML =
        '<div class="v3-pbar-wait">基线采集中…（前 60s 静息阶段结束后显示 ERD%）</div>';
    }
    V3_POWER_ROWS = null;
    V3_POWER_NOTE = null;
  }
  el.v3Panel?.classList.toggle("hidden", !on);
  el.phaseSteps?.classList.toggle("hidden", on || isV2Mode());
  el.phaseStepsV2?.classList.toggle("hidden", true);
  el.phaseStepsV3?.classList.toggle("hidden", !on);
  document.getElementById("st-object-wrap")?.classList.toggle("hidden", on || isV2Mode());
  document.getElementById("st-scene-wrap")?.classList.toggle("hidden", on || isV2Mode());
  document.getElementById("st-v2-round-wrap")?.classList.toggle("hidden", true);
  document.getElementById("st-v2-score-wrap")?.classList.toggle("hidden", true);
  document.getElementById("p4-summary-v1")?.classList.toggle("hidden", on || isV2Mode());
  el.p4SummaryV2?.classList.toggle("hidden", true);
  document.getElementById("btn-gate")?.classList.toggle("hidden", on || isV2Mode());
  document.getElementById("btn-split")?.classList.toggle("hidden", on || isV2Mode());
  if (!on) {
    resetV3EegState();
    if (el.v3FeatureCards) el.v3FeatureCards.innerHTML = "";
    if (el.v3HatCheck) {
      el.v3HatCheck.classList.add("hidden");
      el.v3HatCheck.textContent = "";
    }
    V3_EEG_STATE.hatVerdict = null;
  }
}

function setPhaseStepV3(step) {
  const order = ["baseline", "block", "guidance", "block2", "report"];
  const map = { self_check: "baseline", block_gap: "block", block: "block", report: "report", guidance: "guidance" };
  const key = map[step] || step;
  const idx = order.indexOf(key);
  el.phaseStepsV3?.querySelectorAll("li").forEach((li) => {
    const p = li.getAttribute("data-phase");
    const i = order.indexOf(p);
    li.classList.toggle("active", p === key);
    li.classList.toggle("done", idx >= 0 && i >= 0 && i < idx);
  });
}

function setV2RunPanel(on) {
  if (on) {
    setV3RunPanel(false);
    setV4RunPanel(false);
  }
  el.v2Panel?.classList.toggle("hidden", !on);
  el.phaseSteps?.classList.toggle("hidden", on);
  el.phaseStepsV2?.classList.toggle("hidden", !on);
  el.phaseStepsV3?.classList.toggle("hidden", true);
  document.getElementById("st-object-wrap")?.classList.toggle("hidden", on);
  document.getElementById("st-scene-wrap")?.classList.toggle("hidden", on);
  document.getElementById("st-v2-round-wrap")?.classList.toggle("hidden", !on);
  document.getElementById("st-v2-score-wrap")?.classList.toggle("hidden", !on);
  document.getElementById("p4-summary-v1")?.classList.toggle("hidden", on);
  el.p4SummaryV2?.classList.toggle("hidden", !on);
  document.getElementById("btn-gate")?.classList.toggle("hidden", on);
  document.getElementById("btn-split")?.classList.toggle("hidden", on);
  if (!on) stopGuidanceCountdown();
}

function setPhaseStepV2(step) {
  const order = ["guidance", "calibration", "gate", "game", "end"];
  const key = step === "done" ? "end" : step;
  const idx = order.indexOf(key);
  el.phaseStepsV2?.querySelectorAll("li").forEach((li) => {
    const p = li.getAttribute("data-phase");
    const i = order.indexOf(p);
    li.classList.toggle("active", p === key);
    li.classList.toggle("done", idx >= 0 && i >= 0 && i < idx);
  });
}

function updateV2Progress(prog, score) {
  if (!prog) return;
  if (el.v2CalProg) {
    el.v2CalProg.textContent = `${prog.cal_round ?? 0}/${prog.cal_rounds_max ?? "—"}`;
  }
  if (el.v2GameProg) {
    el.v2GameProg.textContent = `${prog.game_round ?? 0}/${prog.game_rounds ?? "—"}`;
  }
  if (el.v2Subblock) el.v2Subblock.textContent = String(prog.subblock ?? "—");
  if (el.v2FtStatus) el.v2FtStatus.textContent = String(prog.ft_status || "idle");
  if (el.stV2Round) {
    el.stV2Round.textContent =
      `标定 ${prog.cal_round ?? 0}/${prog.cal_rounds_max ?? "—"} · 游戏 ${prog.game_round ?? 0}/${prog.game_rounds ?? "—"}`;
  }
  const sc = score != null ? score : prog.score;
  if (sc != null) {
    const n = Number(sc);
    if (el.stV2Score) el.stV2Score.textContent = n.toFixed(1);
    if (el.v2ScoreNum) el.v2ScoreNum.textContent = n.toFixed(1);
    if (el.v2ScoreFill) el.v2ScoreFill.style.width = `${Math.min(100, (n / 5) * 100)}%`;
  }
  if (prog.phase_step) setPhaseStepV2(prog.phase_step);
}

function updateV2Gate(msg) {
  if (el.v2GateAcc) {
    el.v2GateAcc.textContent =
      msg.acc != null ? `${(Number(msg.acc) * 100).toFixed(1)}%` : "—";
  }
  if (el.v2GateStatus) el.v2GateStatus.textContent = msg.status || "—";
  if (el.v2GateN) el.v2GateN.textContent = String(msg.n_quiz ?? 0);
  if (el.v2GateCurve && Array.isArray(msg.curve)) {
    el.v2GateCurve.textContent = msg.curve.map(([k, a]) => `k=${k}: ${(a * 100).toFixed(1)}%`).join("\n");
  }
  el.v2WeakMi?.classList.toggle("hidden", msg.status !== "weak_mi");
  if (msg.progress) updateV2Progress(msg.progress);
  setPhaseStepV2("gate");
}

el.form.querySelectorAll('input[name="phase_mode"]').forEach((r) => {
  r.addEventListener("change", syncPhaseModeUi);
});
el.form.querySelector('[name="protocol_locked"]')?.addEventListener("change", (ev) => {
  if (!ev.target.checked) {
    const ok = window.confirm("关闭采集冻结锁？时序与高级常量将可改——仅调试用。");
    if (!ok) {
      ev.target.checked = true;
      return;
    }
  }
  syncProtocolLockUi();
});

document.querySelectorAll("[data-v2-timing-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = V2_TIMING_PRESETS[btn.dataset.v2TimingPreset];
    if (!p) return;
    for (const [k, v] of Object.entries(p)) {
      const node = el.form.elements.namedItem(`v2_${k}`);
      if (node) node.value = v;
    }
    renderSetupTimelineV2();
  });
});

const TIMING_KEYS = [
  ["t_fixation_s", "fixation_s", "注视", "tl-fixation"],
  ["t_cue_s", "cue_s", "提示", "tl-cue"],
  ["t_mi_s", "mi_s", "MI", "tl-mi"],
  ["t_post_mi_hold_s", "post_mi_hold_s", "保持", "tl-hold"],
  ["t_rest_s", "rest_s", "静息", "tl-rest"],
  ["t_transition_s", "transition_s", "过渡", "tl-transition"],
];

const TIMING_PRESETS = {
  standard: { fixation_s: 2, cue_s: 2, mi_s: 4, post_mi_hold_s: 1, rest_s: 4, transition_s: 3 },
  mi6: { fixation_s: 2, cue_s: 2, mi_s: 6, post_mi_hold_s: 1, rest_s: 4, transition_s: 3 },
  mi8: { fixation_s: 2, cue_s: 2, mi_s: 8, post_mi_hold_s: 1, rest_s: 4, transition_s: 3 },
};

function readTimingFromForm() {
  const t = {};
  for (const [formName, cfgKey] of TIMING_KEYS) {
    const node = el.form.elements.namedItem(formName);
    const v = Number(node && node.value);
    t[cfgKey] = Number.isFinite(v) ? v : 0;
  }
  return t;
}

function setTimingToForm(timing) {
  for (const [formName, cfgKey] of TIMING_KEYS) {
    const node = el.form.elements.namedItem(formName);
    if (node && timing && timing[cfgKey] != null) node.value = timing[cfgKey];
  }
  renderSetupTimeline();
}

/** 渲染时间轴：按各阶段秒数比例着色分段；container 为空则跳过 */
function renderTimeline(container, timing) {
  if (!container) return null;
  const order = TIMING_KEYS.map(([, key, zh, cls]) => ({ key, zh, cls, s: Number(timing[key]) || 0 }));
  const total = order.reduce((a, b) => a + b.s, 0);
  container.innerHTML = "";
  if (total <= 0) return { total };
  for (const seg of order) {
    if (seg.s <= 0) continue;
    const div = document.createElement("div");
    div.className = `tl-seg ${seg.cls}`;
    div.style.flexGrow = String(seg.s);
    div.title = `${seg.zh} ${seg.s}s`;
    const label = document.createElement("span");
    label.textContent = `${seg.zh} ${seg.s}s`;
    div.appendChild(label);
    container.appendChild(div);
  }
  const cap = document.createElement("div");
  cap.className = "tl-total";
  cap.textContent = `单 trial 合计 ${total}s`;
  container.appendChild(cap);
  return { total };
}

function updateTimingHint(total) {
  if (!el.timingHint) return;
  const fd = new FormData(el.form);
  const acquire = Number(fd.get("acquire_trials") || 0) || 0;
  const learnN = Number(fd.get("learn_trials_per_step") || 0) || 0;
  const skipAdapt = el.form.querySelector('[name="skip_adapt"]')?.checked;
  const skipLearn = el.form.querySelector('[name="skip_learn"]')?.checked;
  let trials = acquire;
  if (!skipLearn) trials += 3 * learnN;
  if (!skipAdapt) trials += 2;
  const est = trials * total;
  el.timingHint.textContent =
    `单 trial = ${total}s · 预计试次总数 ≈ ${trials}` +
    `（正式 ${acquire}${skipLearn ? "" : ` + 学习 ${3 * learnN}`}${skipAdapt ? "" : " + 适应 2"}）` +
    ` · 纯试次时长 ≈ ${Math.round(est / 60)} 分钟（不含弹窗确认与暂停）`;
}

function renderSetupTimeline() {
  const { total } = renderTimeline(el.setupTimeline, readTimingFromForm()) || { total: 0 };
  updateTimingHint(total);
}

function applyTimingPreset(name) {
  const preset = TIMING_PRESETS[name];
  if (preset) setTimingToForm(preset);
}

document.querySelectorAll("[data-timing-preset]").forEach((btn) => {
  btn.addEventListener("click", () => applyTimingPreset(btn.dataset.timingPreset));
});

let ws = null;
let subjectUrl = `http://${location.hostname || "127.0.0.1"}:8080/`;
let sessionRoot = "";
let defaultsFromServer = null;
let builtinDefaults = null;
let paused = false;
let starting = false;
let restartAfterAbort = false;
let hotkeysEnabled = true;
let lockedConfig = null;

function showView(name) {
  el.setup.classList.toggle("hidden", name !== "setup");
  el.run.classList.toggle("hidden", name !== "run");
  el.summary.classList.toggle("hidden", name !== "summary");
  location.hash = name;
}

/** 清空运行页残留（再开一场 / 新开 session 前必须调用） */
function resetRunView() {
  stopGuidanceCountdown();
  paused = false;
  const btnPause = document.getElementById("btn-pause");
  if (btnPause) btnPause.textContent = "暂停";

  showRunAlert("");
  if (el.popupWarn) el.popupWarn.classList.add("hidden");
  if (el.runSummary) el.runSummary.innerHTML = "";
  if (el.runTimeline) el.runTimeline.innerHTML = "";
  if (el.runTimingHint) el.runTimingHint.textContent = "";
  const qRun = document.getElementById("q-status-run");
  if (qRun) qRun.textContent = "";

  if (el.stPhase) el.stPhase.textContent = "—";
  if (el.stStage) el.stStage.textContent = "—";
  if (el.stTrial) el.stTrial.textContent = "—";
  if (el.stLabel) el.stLabel.textContent = "—";
  if (el.stObject) el.stObject.textContent = "—";
  if (el.stScene) el.stScene.textContent = "—";
  if (el.stReject) el.stReject.textContent = "0";
  if (el.stAcq) el.stAcq.textContent = "—";
  if (el.linkLine) {
    el.linkLine.textContent = "—";
    el.linkLine.className = "stat-ok";
    el.linkLine.title = "";
  }

  setPhaseStep("adapt");
  setPhaseStepV2("guidance");
  setPhaseStepV3("baseline");

  resetV3EegState();
  if (el.v3FeatureCards) el.v3FeatureCards.innerHTML = "";
  if (el.v3HatCheck) {
    el.v3HatCheck.textContent = "";
    el.v3HatCheck.className = "v3-hat-check hidden";
  }
  if (el.v3PowerBars) {
    el.v3PowerBars.innerHTML =
      '<div class="v3-pbar-wait">基线采集中…（前 60s 静息阶段结束后显示 ERD%）</div>';
  }
  if (el.v3BlockProg) el.v3BlockProg.textContent = "—";
  if (el.v3TrialProg) el.v3TrialProg.textContent = "—";
  if (el.v3Cond) el.v3Cond.textContent = "—";
  if (el.v3BlockNvalid) el.v3BlockNvalid.textContent = "—";
  if (el.v3BlockNlr) el.v3BlockNlr.textContent = "—";
  if (el.v3BlockAcc) el.v3BlockAcc.textContent = "—";
  if (el.v3BlockErd) el.v3BlockErd.textContent = "—";
  if (el.v3BlockLat) el.v3BlockLat.textContent = "—";
  if (el.v3Eeg) {
    const ctx = el.v3Eeg.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, el.v3Eeg.width, el.v3Eeg.height);
  }
  if (el.v3EegGain) el.v3EegGain.textContent = "±— µV";
  el.btnV3Guidance?.classList.add("hidden");
  if (el.btnV3Guidance) el.btnV3Guidance.disabled = false;

  resetV4State();
  if (el.v4Metrics) el.v4Metrics.innerHTML = "";
  if (el.v4Channels) el.v4Channels.innerHTML = "";
  if (el.v4Problems) el.v4Problems.innerHTML = "";
  if (el.v4Headline) el.v4Headline.textContent = "";
  if (el.v4VerdictText) el.v4VerdictText.textContent = "—";
  if (el.v4Streak) el.v4Streak.textContent = "0/5";
  if (el.v4StreakFill) el.v4StreakFill.style.width = "0%";
  if (el.btnV4ToV3) el.btnV4ToV3.disabled = true;
  el.v4Panel?.classList.add("hidden");
  el.phaseStepsV4?.classList.add("hidden");
  setPhaseStepV4("check");

  if (el.v2StageHint) el.v2StageHint.textContent = "等待 v2 阶段消息…";
  if (el.v2CalProg) el.v2CalProg.textContent = "—";
  if (el.v2GameProg) el.v2GameProg.textContent = "—";
  if (el.v2Subblock) el.v2Subblock.textContent = "—";
  if (el.v2FtStatus) el.v2FtStatus.textContent = "idle";
  if (el.v2ScoreNum) el.v2ScoreNum.textContent = "—";
  if (el.v2ScoreFill) el.v2ScoreFill.style.width = "0%";
  if (el.v2GateAcc) el.v2GateAcc.textContent = "—";
  if (el.v2GateStatus) el.v2GateStatus.textContent = "—";
  if (el.v2GateN) el.v2GateN.textContent = "0";
  if (el.v2GateCurve) el.v2GateCurve.textContent = "—";
  el.v2WeakMi?.classList.add("hidden");
  if (el.btnV2Guidance) el.btnV2Guidance.disabled = false;
  if (el.stV2Round) el.stV2Round.textContent = "—";
  if (el.stV2Score) el.stV2Score.textContent = "—";
}

function setWsStatus(text, cls) {
  el.wsStatus.textContent = text;
  el.wsStatus.className = "ws-status" + (cls ? ` ${cls}` : "");
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

function formToRunConfig() {
  const fd = new FormData(el.form);
  const seedRaw = String(fd.get("seed") || "").trim();
  const board =
    fd.get("board_mode") ||
    (el.form.querySelector('input[name="board_mode"]:checked') || {}).value ||
    "synthetic";
  const acqEnabled = el.form.querySelector('[name="acq_enabled"]').checked;
  const layout = fd.get("save_layout") || "phase_folders";
  return {
    schema_version: 2,
    subject: {
      subject_id: String(fd.get("subject_id") || "").trim(),
      session_id: String(fd.get("session_id") || "").trim(),
      notes: String(fd.get("notes") || ""),
    },
    acquisition: {
      enabled: acqEnabled,
      board_mode: board,
      serial_port: String(fd.get("serial_port") || "COM5").trim(),
      sample_rate_hz: 250,
      markers_lsl: acqEnabled,
      filter: {
        enabled: el.form.querySelector('[name="filter_enabled"]')?.checked !== false,
        bandpass_low_hz: Number(fd.get("bandpass_low_hz") || 0.5),
        bandpass_high_hz: Number(fd.get("bandpass_high_hz") || 45),
        notch_low_hz: Number(fd.get("notch_low_hz") || 49),
        notch_high_hz: Number(fd.get("notch_high_hz") || 51),
      },
    },
    experiment: {
      phase_mode: String(fd.get("phase_mode") || "phase2_full"),
      acquire_trials: Number(fd.get("acquire_trials") || 40),
      learn_trials_per_step: Number(fd.get("learn_trials_per_step") || 2),
      seed: (() => {
        const v2Seed = String(fd.get("v2_seed") || "").trim();
        const v3Seed = String(fd.get("v3_seed") || "").trim();
        const v1Seed = seedRaw;
        if (isV3Mode()) return v3Seed === "" ? null : Number(v3Seed);
        if (isV2Mode()) return v2Seed === "" ? null : Number(v2Seed);
        return v1Seed === "" ? null : Number(v1Seed);
      })(),
      open_subject_page: isV4Mode() ? false : el.form.querySelector('[name="open_subject_page"]').checked,
      skip_adapt: el.form.querySelector('[name="skip_adapt"]')?.checked || false,
      skip_learn: el.form.querySelector('[name="skip_learn"]')?.checked || false,
      skip_gate: el.form.querySelector('[name="skip_gate"]')?.checked || false,
      protocol_locked: el.form.querySelector('[name="protocol_locked"]')?.checked !== false,
      v2_overrides: readV2Overrides(),
      v3_overrides: readV3Overrides(),
      skip_v2_guidance: el.form.querySelector('[name="skip_v2_guidance"]')?.checked || false,
      skip_v2_calibration: el.form.querySelector('[name="skip_v2_calibration"]')?.checked || false,
      skip_v2_gate: el.form.querySelector('[name="skip_v2_gate"]')?.checked || false,
      skip_v2_game: el.form.querySelector('[name="skip_v2_game"]')?.checked || false,
      ready_timeout_s: 90,
      timing: readTimingFromForm(),
      phase4: {
        window_mode: el.form.elements.namedItem("p4_window_mode")?.value || "fixed",
        win_sec: Number(el.form.elements.namedItem("p4_win_sec")?.value) || 2,
        hop_ms: Number(el.form.elements.namedItem("p4_hop_ms")?.value) || 100,
      },
    },
    storage: {
      save_root: String(fd.get("save_root") || "experiment_game/data/sessions").trim(),
      save_layout: layout,
      save_eeg: acqEnabled,
      save_events: true,
      save_session_meta: true,
      save_continuous_master: layout === "phase_folders",
      save_phase_slices: layout === "phase_folders",
      save_trial_index: true,
      auto_phase4: el.form.querySelector('[name="auto_phase4"]')?.checked || false,
    },
    ui: {
      remember_last_config: el.form.querySelector('[name="remember_last_config"]')?.checked !== false,
      skip_setup_if_unchanged: el.form.querySelector('[name="skip_setup_if_unchanged"]')?.checked || false,
      operator_hotkeys: el.form.querySelector('[name="operator_hotkeys"]')?.checked !== false,
    },
    extensions: {},
  };
}

function applyConfigToForm(cfg) {
  if (!cfg) return;
  const set = (name, value) => {
    const node = el.form.elements.namedItem(name);
    if (!node) return;
    if (node instanceof RadioNodeList) {
      for (const r of node) {
        if (r.value === String(value)) r.checked = true;
      }
      return;
    }
    if (node.type === "checkbox") {
      node.checked = Boolean(value);
      return;
    }
    node.value = value == null ? "" : String(value);
  };
  set("subject_id", cfg.subject?.subject_id);
  set("session_id", cfg.subject?.session_id);
  set("notes", cfg.subject?.notes || "");
  set("open_subject_page", cfg.experiment?.open_subject_page !== false);
  set("acq_enabled", cfg.acquisition?.enabled !== false);
  set("board_mode", cfg.acquisition?.board_mode || "synthetic");
  set("serial_port", cfg.acquisition?.serial_port || "COM5");
  set("acquire_trials", cfg.experiment?.acquire_trials ?? 40);
  set("learn_trials_per_step", cfg.experiment?.learn_trials_per_step ?? 2);
  set("seed", cfg.experiment?.seed ?? "");
  set("skip_adapt", cfg.experiment?.skip_adapt);
  set("skip_learn", cfg.experiment?.skip_learn);
  set("skip_gate", cfg.experiment?.skip_gate);
  set("protocol_locked", cfg.experiment?.protocol_locked !== false);
  set("skip_v2_guidance", cfg.experiment?.skip_v2_guidance);
  set("skip_v2_calibration", cfg.experiment?.skip_v2_calibration);
  set("skip_v2_gate", cfg.experiment?.skip_v2_gate);
  set("skip_v2_game", cfg.experiment?.skip_v2_game);
  set("v2_seed", cfg.experiment?.seed ?? "");
  applyV2OverridesToForm(cfg.experiment?.v2_overrides || {});
  const pm = cfg.experiment?.phase_mode || "phase2_full";
  for (const r of el.form.querySelectorAll('input[name="phase_mode"]')) {
    r.checked = r.value === pm;
  }
  syncPhaseModeUi();
  syncProtocolLockUi();
  set("save_root", cfg.storage?.save_root);
  set("save_layout", cfg.storage?.save_layout || "phase_folders");
  set("auto_phase4", cfg.storage?.auto_phase4);
  set("remember_last_config", cfg.ui?.remember_last_config !== false);
  set("skip_setup_if_unchanged", cfg.ui?.skip_setup_if_unchanged);
  set("operator_hotkeys", cfg.ui?.operator_hotkeys !== false);
  const filt = cfg.acquisition?.filter || {};
  set("filter_enabled", filt.enabled !== false);
  set("bandpass_low_hz", filt.bandpass_low_hz ?? 0.5);
  set("bandpass_high_hz", filt.bandpass_high_hz ?? 45);
  set("notch_low_hz", filt.notch_low_hz ?? 49);
  set("notch_high_hz", filt.notch_high_hz ?? 51);
  setTimingToForm(cfg.experiment?.timing || {});
  const p4 = cfg.experiment?.phase4 || {};
  const setSel = (name, value) => {
    const node = el.form.elements.namedItem(name);
    if (node) node.value = value;
  };
  setSel("p4_window_mode", p4.window_mode || "fixed");
  set("p4_win_sec", p4.win_sec ?? 2);
  set("p4_hop_ms", p4.hop_ms ?? 100);
  // Summary 页手动切窗控件默认跟随配置
  const runMode = document.getElementById("p4-run-mode");
  const runWin = document.getElementById("p4-run-win");
  const runHop = document.getElementById("p4-run-hop");
  if (runMode) runMode.value = p4.window_mode || "fixed";
  if (runWin) runWin.value = p4.win_sec ?? 2;
  if (runHop) runHop.value = p4.hop_ms ?? 100;
  hotkeysEnabled = cfg.ui?.operator_hotkeys !== false;
  syncAcqUi();
}

function syncAcqUi() {
  const acqOn = el.form.querySelector('[name="acq_enabled"]').checked;
  const cyton = el.form.querySelector('input[name="board_mode"]:checked')?.value === "cyton";
  el.acqWarn.classList.toggle("hidden", acqOn);
  el.guiHint.classList.toggle("hidden", !cyton);
  el.deviceFs.disabled = !acqOn || !cyton;
  const layout = el.form.querySelector('[name="save_layout"]')?.value || "flat";
  if (el.saveHint) {
    el.saveHint.textContent = acqOn
      ? layout === "phase_folders"
        ? "将写入 continuous/ + by_phase/ + alignment/（EEG 与 Marker 同一 LSL 时钟）"
        : "扁平落盘：会话根 eeg.csv + events.jsonl + session.meta.json；并写 alignment/"
      : "仅 events + meta，无脑电，不能 Phase4 训练";
  }
}

function showErrors(list) {
  if (!list || !list.length) {
    el.errors.classList.add("hidden");
    el.errors.textContent = "";
    return;
  }
  el.errors.classList.remove("hidden");
  el.errors.textContent = list.join("\n");
}

function setPhaseStep(phase) {
  const map = { adapt: "adapt", learn: "learn", gate: "gate", acquire: "acquire", end: "end", done: "end" };
  const key = map[phase] || phase;
  const order = ["adapt", "learn", "gate", "acquire", "end"];
  const idx = order.indexOf(key);
  el.phaseSteps.querySelectorAll("li").forEach((li) => {
    const p = li.getAttribute("data-phase");
    const i = order.indexOf(p);
    li.classList.toggle("active", p === key);
    li.classList.toggle("done", idx >= 0 && i >= 0 && i < idx);
  });
}

function configBrief(cfg) {
  if (!cfg) return "";
  const acq = cfg.acquisition || {};
  const exp = cfg.experiment || {};
  const parts = [
    `${cfg.subject?.subject_id}/${cfg.subject?.session_id}`,
    acq.enabled ? (acq.board_mode === "cyton" ? `真机 ${acq.serial_port}` : "合成板") : "不采数",
    `trials=${exp.acquire_trials}`,
    `MI=${exp.timing?.mi_s ?? 4}s`,
    cfg.storage?.save_layout || "phase_folders",
  ];
  return parts.join(" · ");
}

function fillSerialPorts(ports) {
  if (!el.portList) return;
  el.portList.innerHTML = "";
  const list = ports || [];
  for (const p of list) {
    const opt = document.createElement("option");
    opt.value = p.device;
    opt.label = p.description && p.description !== p.device ? `${p.device} — ${p.description}` : p.device;
    el.portList.appendChild(opt);
  }
  if (el.portsHint) {
    el.portsHint.textContent = list.length
      ? `已枚举 ${list.length} 个串口；可点选或手填。`
      : "未枚举到串口；请手填 COM（并确认设备已连接）。";
  }
}

function updateRunLockSummary(msg) {
  lockedConfig = {
    acq_enabled: msg.acq_enabled,
    board_mode: msg.board_mode,
    serial_port: msg.serial_port,
    acquire_trials: msg.acquire_trials,
    save_root: msg.save_root,
    session_root: msg.session_root,
  };
  el.runSummary.innerHTML = msg.phase_mode === "v4_session"
    ? [
        `<div><span class="k">模式</span>v4 质量检测</div>`,
        `<div><span class="k">采集</span>${msg.acq_enabled ? "开" : "关"}</div>`,
        `<div><span class="k">最长</span>${msg.v4_config_effective?.duration_s ?? 90}s · 连续 ${msg.v4_config_effective?.pass_streak_required ?? 5} 窗</div>`,
        `<div><span class="k">会话目录</span><code>${msg.session_root || "—"}</code></div>`,
      ].join("")
    : msg.phase_mode === "v3_session"
    ? [
        `<div><span class="k">模式</span>v3 探针${msg.protocol_locked ? " · 冻结" : " · 调试"}</div>`,
        `<div><span class="k">块顺序</span>${(msg.v3_block_order || []).join(" → ") || "—"}</div>`,
        `<div><span class="k">采集</span>${msg.acq_enabled ? "开" : "关"}</div>`,
        `<div><span class="k">会话目录</span><code>${msg.session_root || "—"}</code></div>`,
      ].join("")
    : msg.phase_mode === "v2_session"
    ? [
        `<div><span class="k">模式</span>v2 会话${msg.protocol_locked ? " · 冻结" : " · 调试"}</div>`,
        `<div><span class="k">采集</span>${msg.acq_enabled ? "开" : "关"}</div>`,
        `<div><span class="k">轮数</span>标定 ${msg.v2_config_effective?.cal_rounds_min ?? "?"}–${msg.v2_config_effective?.cal_rounds_max ?? "?"} / 游戏 ${msg.v2_config_effective?.game_rounds ?? "?"}</div>`,
        `<div><span class="k">会话目录</span><code>${msg.session_root || "—"}</code></div>`,
      ].join("")
    : [
        `<div><span class="k">采集</span>${msg.acq_enabled ? "开" : "关"}</div>`,
        `<div><span class="k">板卡</span>${msg.board_mode}${msg.board_mode === "cyton" ? " / " + (msg.serial_port || "") : ""}</div>`,
        `<div><span class="k">正式 trials</span>${msg.acquire_trials}</div>`,
        `<div><span class="k">会话目录</span><code>${msg.session_root || "—"}</code></div>`,
      ].join("");
  // 本场锁定的时序构成
  if (msg.timing) {
    renderTimeline(el.runTimeline, msg.timing);
    if (el.runTimingHint) {
      if (msg.phase_mode === "v3_session") {
        el.runTimingHint.textContent =
          `prep ${msg.timing.fixation_s ?? msg.timing.prep_s}s → cue ${msg.timing.cue_s}s → ` +
          `MI ${msg.timing.mi_s}s → ITI ${msg.timing.transition_s ?? msg.timing.iti_s}s` +
          (msg.trial_total_s ? ` · 合计 ${msg.trial_total_s}s` : "");
      } else if (msg.phase_mode === "v2_session") {
        el.runTimingHint.textContent =
          `prep ${msg.timing.fixation_s ?? msg.timing.prep_s}s → cue ${msg.timing.cue_s}s → ` +
          `MI ${msg.timing.mi_s}s → ITI ${msg.timing.transition_s ?? msg.timing.iti_s}s` +
          (msg.trial_total_s ? ` · 合计 ${msg.trial_total_s}s` : "");
      } else {
        el.runTimingHint.textContent =
          `注视 ${msg.timing.fixation_s}s → 提示 ${msg.timing.cue_s}s → ` +
          `MI ${msg.timing.mi_s}s → 保持 ${msg.timing.post_mi_hold_s}s → ` +
          `静息 ${msg.timing.rest_s}s → 过渡 ${msg.timing.transition_s}s` +
          (msg.trial_total_s ? ` · 合计 ${msg.trial_total_s}s` : "");
      }
    }
  }
}

function maybeShowReuseBar(cfg) {
  if (!el.reuseBar) return;
  const want = Boolean(cfg?.ui?.skip_setup_if_unchanged);
  if (!want || !cfg) {
    el.reuseBar.classList.add("hidden");
    return;
  }
  el.reuseSummary.textContent = configBrief(cfg);
  el.reuseBar.classList.remove("hidden");
}

function connect() {
  setWsStatus("连接中…");
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    setWsStatus("已连接", "ok");
    send({ type: "operator_hello" });
  };
  ws.onclose = () => {
    setWsStatus("已断开，重连中…", "err");
    setTimeout(connect, 1200);
  };
  ws.onerror = () => setWsStatus("WebSocket 错误", "err");
  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    handleMessage(msg);
  };
}

function showPhase4Result(p4) {
  if (!el.phase4Msg) return;
  if (!p4) {
    el.phase4Msg.textContent =
      "可点「一键 Phase4 切窗」：仅 acquire + 未 reject → data/epochs/<会话名>/";
    return;
  }
  const s = p4.summary || {};
  if (p4.ok) {
    const w = s.window || {};
    const wdesc = w.mode === "slide"
      ? `滑动窗 ${w.win_sec}s/步长${w.hop_ms}ms`
      : `固定窗 ${w.win_sec ?? 2}s`;
    el.phase4Msg.textContent =
      `Phase4 OK · ${wdesc} · N=${s.n ?? "—"} · X=${JSON.stringify(s.X_shape || [])} · ` +
      `y_task=${JSON.stringify(s.y_task_counts || {})} → ${p4.epochs_dir || ""}`;
  } else {
    el.phase4Msg.textContent = `Phase4 失败：${p4.message || "未知错误"}`;
  }
}

function handleMessage(msg) {
  const t = msg.type;
  if (t === "operator_hello") {
    if (msg.subject_url) subjectUrl = msg.subject_url;
    defaultsFromServer = msg.defaults || null;
    builtinDefaults = msg.builtin_defaults || defaultsFromServer;
    const local = loadLocalDefaults();
    // 优先：服务端文件默认 > 浏览器 localStorage > 内置
    applyConfigToForm(defaultsFromServer || local || builtinDefaults);
    fillSerialPorts(msg.serial_ports || []);
    maybeShowReuseBar(defaultsFromServer || local);
    if (msg.defaults_warning) showErrors([msg.defaults_warning]);
  } else if (t === "serial_ports") {
    fillSerialPorts(msg.ports || []);
    if (!msg.ok && msg.message) showErrors([msg.message]);
  } else if (t === "save_defaults_ack") {
    if (msg.ok) {
      if (msg.run_config) {
        saveLocalDefaults(msg.run_config);
        defaultsFromServer = msg.run_config;
      }
      alert(`已保存默认配置\n${msg.path || msg.message || ""}`);
    } else {
      showErrors([msg.message || "保存默认失败"]);
    }
  } else if (t === "config_ack") {
    if (!msg.ok) {
      starting = false;
      showErrors(msg.errors || ["配置无效"]);
      showView("setup");
      return;
    }
    showErrors([]);
    if (msg.starting) {
      starting = true;
      if (el.reuseBar) el.reuseBar.classList.add("hidden");
      showView("run");
      if (!isV4Mode()) el.popupWarn.classList.remove("hidden");
      if (!isV4Mode()) tryOpenSubject();
    }
  } else if (t === "session_started") {
    if (msg.subject_url) subjectUrl = msg.subject_url;
    sessionRoot = msg.session_root || "";
    const v2 = msg.phase_mode === "v2_session";
    const v3 = msg.phase_mode === "v3_session";
    const v4 = msg.phase_mode === "v4_session";
    setV2RunPanel(v2);
    setV3RunPanel(v3);
    setV4RunPanel(v4);
    if (v4) {
      showRunAlert("");
      onV4Start(msg);
      if (el.stPhase) el.stPhase.textContent = "v4";
      if (el.stStage) el.stStage.textContent = "质量检测";
    } else if (v3) {
      showRunAlert("");
      setPhaseStepV3("baseline");
      if (msg.v3_block_order) {
        updateV3Block({
          block_idx: 0,
          blocks_total: msg.v3_config_effective?.blocks ?? 2,
          trials_per_block: msg.v3_config_effective?.trials_per_block ?? 18,
          // 基线阶段尚未进块，条件列不提前显示第一块条件
          cond: "基线",
        });
      }
    } else if (v2 && msg.degraded) {
      showRunAlert("演练模式：权重不可用，无在线判定/微调", "degraded");
    } else if (v2 && msg.degraded_pending_lsl) {
      showRunAlert("采集启动中，正在挂接 LSL 在线判定…", "degraded");
    } else {
      showRunAlert("");
    }
    if (v2 && el.v2StageHint) {
      el.v2StageHint.textContent = "v2 会话已启动；轮间请点「确认动觉引导完成」";
      setPhaseStepV2("guidance");
    }
    updateRunLockSummary(msg);
    // 诱导页已在 config_ack 打开；此处不再重复（避免多窗口）
  } else if (t === "v4_start") {
    onV4Start(msg);
  } else if (t === "v4_live") {
    onV4Live(msg);
  } else if (t === "v4_quality") {
    updateV4Quality(msg);
  } else if (t === "v4_pass") {
    onV4Pass(msg);
    updateV4Quality({ ...msg, rolling_verdict: "pass", pass_streak: msg.streak });
  } else if (t === "v4_summary") {
    onV4Summary(msg);
  } else if (t === "eeg_frame") {
    if (isV3Mode() || !el.v3Panel?.classList.contains("hidden")) drawV3EegFrame(msg);
  } else if (t === "v3_baseline") {
    handleV3Baseline(msg);
    renderV3PowerBars(msg);
  } else if (t === "trial_features") {
    appendV3FeatureCard(msg);
  } else if (t === "v3_block") {
    updateV3Block(msg);
    if (msg.phase === "begin") setPhaseStepV3("block");
  } else if (t === "v3_report") {
    setPhaseStepV3("report");
    if (el.v3SummaryDetail) {
      el.v3SummaryDetail.classList.remove("hidden");
      const r = msg.report || {};
      el.v3SummaryDetail.textContent =
        `v3 报告 · 质量=${r.quality_tier || "—"} · frozen=${r.frozen}`;
    }
  } else if (t === "v3_abort") {
    showRunAlert(`v3 已中止：${msg.reason || "operator_abort"}`, "abort");
    stopGuidanceCountdown();
  } else if (t === "v3_warn") {
    showRunAlert(msg.message || "v3 链路警告", "degraded");
  } else if (t === "v2_online_status") {
    if (msg.degraded) {
      showRunAlert(msg.message || "演练模式：无在线判定/微调", "degraded");
    } else {
      showRunAlert("");
      if (el.v2StageHint) el.v2StageHint.textContent = msg.message || "在线判定与微调已启用";
    }
  } else if (t === "v2_abort") {
    const reason = msg.reason || "unknown";
    const kind = msg.kind === "guidance_timeout" ? "guidance_timeout" : reason;
    showRunAlert(`v2 中止/熔断：${kind}${msg.consecutive_invalid != null ? `（连续无效 ${msg.consecutive_invalid}）` : ""}`, "abort");
    playAlertBeep("alert");
    stopGuidanceCountdown();
    if (el.v2StageHint) el.v2StageHint.textContent = `v2 已中止 · ${reason}`;
  } else if (t === "v2_gate") {
    updateV2Gate(msg);
  } else if (t === "v2_stage") {
    const v3Active = !el.v3Panel?.classList.contains("hidden");
    if (v3Active) {
      markV3StageEvent(msg.stage);
      el.stPhase.textContent = msg.progress?.phase_step || msg.stage || "—";
      el.stStage.textContent = msg.stage || "—";
      if (msg.ctx?.trial_id != null) el.stTrial.textContent = msg.ctx.trial_id;
      if (msg.ctx?.label != null) el.stLabel.textContent = msg.ctx.label;
      if (msg.progress?.phase_step) setPhaseStepV3(msg.progress.phase_step);
      if (msg.stage === "guidance_begin") {
        el.btnV3Guidance?.classList.remove("hidden");
        if (el.btnV3Guidance) el.btnV3Guidance.disabled = false;
        startGuidanceCountdown(msg.data?.timeout_s ?? 600);
      }
      if (msg.stage === "guidance_end") {
        el.btnV3Guidance?.classList.add("hidden");
        stopGuidanceCountdown();
      }
    } else {
      if (el.v2StageHint) {
        const stage = msg.stage || "—";
        const mode = msg.ctx?.mode || msg.data?.mode || "";
        el.v2StageHint.textContent = `v2 · ${stage}${mode ? ` (${mode})` : ""}`;
      }
      el.stPhase.textContent = msg.progress?.phase_step || msg.stage || "—";
      el.stStage.textContent = msg.stage || "—";
      if (msg.ctx?.trial_id != null) el.stTrial.textContent = msg.ctx.trial_id;
      if (msg.ctx?.label != null) el.stLabel.textContent = msg.ctx.label;
      updateV2Progress(msg.progress, msg.score);
      if (msg.stage === "guidance_begin" && el.btnV2Guidance) {
        el.btnV2Guidance.disabled = false;
        const timeout = msg.data?.timeout_s ?? msg.data?.gap_s ?? (msg.data?.round === 0 ? 600 : 180);
        startGuidanceCountdown(timeout);
      }
      if (msg.stage === "guidance_end" || msg.stage === "gate_pass") {
        stopGuidanceCountdown();
        if (el.btnV2Guidance) el.btnV2Guidance.disabled = true;
      }
      if (msg.stage === "gate_pass") {
        showRunAlert("准入通过，即将进入游戏环节", "gate-pass");
        playAlertBeep("guidance");
        setPhaseStepV2("game");
      }
      if (msg.stage === "weak_mi") {
        showRunAlert("weak_mi：标定未完全达标，仍将进入游戏（全程标记）", "degraded");
        el.v2WeakMi?.classList.remove("hidden");
      }
    }
  } else if (t === "acq_status") {
    el.stAcq.textContent = `${msg.state || "—"}${msg.message ? " · " + msg.message : ""}`;
  } else if (t === "link_status") {
    updateLinkLine(msg);
  } else if (t === "stage") {
    el.stPhase.textContent = msg.phase || "—";
    el.stStage.textContent = msg.stage || "—";
    el.stTrial.textContent = msg.trial_id ?? "—";
    el.stLabel.textContent = msg.label ?? "—";
    el.stObject.textContent = msg.object || "—";
    el.stScene.textContent = msg.scene || "—";
    if (msg.phase) setPhaseStep(msg.phase === "waiting_ready" ? "adapt" : msg.phase);
  } else if (t === "session") {
    if (msg.status === "aborting") {
      showRunAlert(msg.message || "正在中止会话…", "abort");
      stopGuidanceCountdown();
    }
    if (msg.phase) {
      el.stPhase.textContent = msg.phase;
      setPhaseStep(msg.status === "done" ? "end" : msg.phase);
    }
    if (msg.status === "error") {
      el.stAcq.textContent = `错误: ${msg.message || ""}`;
      showErrors([msg.message || "会话错误"]);
    }
    if (msg.status === "done") setPhaseStep("end");
  } else if (t === "operator_state") {
    paused = Boolean(msg.paused);
    el.stReject.textContent = String(msg.reject_count ?? 0);
    document.getElementById("btn-pause").textContent = paused ? "继续" : "暂停";
  } else if (t === "session_segment_saved") {
    // 换场中场：本场已落盘，等待第二次 B；停留在运行页并提示
    if (el.runSummary) {
      el.runSummary.innerHTML = [
        `<div><span class="k">已保存</span><code>${msg.root || "—"}</code></div>`,
        `<div><span class="k">本段 trial</span>${msg.trials_done ?? "—"}</div>`,
        `<div><span class="k">剩余</span>${msg.trials_remaining ?? "—"} 个 trial</div>`,
        `<div><span class="k">下一步</span>引导抬手休息后按 B 开下一段</div>`,
      ].join("");
    }
  } else if (t === "session_saved") {
    starting = false;
    if (restartAfterAbort) {
      restartAfterAbort = false;
      resetRunView();
      lockedConfig = null;
      sessionRoot = "";
      showView("setup");
      maybeShowReuseBar(defaultsFromServer || loadLocalDefaults());
      return;
    }
    showView("summary");
    sessionRoot = msg.root || sessionRoot;
    el.summaryRoot.textContent = sessionRoot || "—";
    el.summaryMsg.textContent = msg.message || "会话已结束";
    el.summaryFiles.innerHTML = "";
    for (const f of msg.files || []) {
      const li = document.createElement("li");
      li.className = "ok";
      li.textContent = f;
      el.summaryFiles.appendChild(li);
    }
    const badge = el.verifyBadge;
    if (badge) {
      const acc = msg.v2_acceptance;
      if (msg.phase_mode === "v2_session" && acc) {
        const map = { green: ["v2 PASS", "pass"], yellow: ["v2 部分", "warn"], red: ["v2 FAIL", "fail"], na: ["v2 演练", "na"] };
        const [txt, cls] = map[acc.level] || ["v2 —", "na"];
        badge.textContent = txt;
        badge.className = cls;
        if (el.v2AcceptDetail) {
          el.v2AcceptDetail.textContent = (acc.gates || [])
            .map((g) => `${g.ok ? "✓" : "✗"} ${g.detail}`)
            .join(" · ");
        }
      } else if (msg.phase_mode === "v3_session" && msg.v3_summary) {
        badge.textContent = `v3 ${msg.v3_summary.quality_tier || "—"}`;
        badge.className = msg.v3_summary.frozen ? "pass" : "warn";
        if (el.v3SummaryDetail) {
          el.v3SummaryDetail.classList.remove("hidden");
          const r = msg.v3_summary.report || {};
          el.v3SummaryDetail.textContent =
            `块顺序 ${(msg.v3_summary.block_order || []).join("→")} · frozen=${msg.v3_summary.frozen} · ${r.quality_tier || ""}`;
        }
      } else if (msg.phase_mode === "v4_session" && msg.v4_summary) {
        const v = msg.v4_summary.verdict || "—";
        badge.textContent = `v4 ${String(v).toUpperCase()}`;
        badge.className = v === "pass" ? "pass" : v === "warn" ? "warn" : "fail";
        if (el.v4SummaryDetail) {
          el.v4SummaryDetail.classList.remove("hidden");
          el.v4SummaryDetail.textContent = msg.v4_summary.recommendation || "";
        }
      } else if (msg.verify && msg.verify.passed === true) {
        badge.textContent = "对齐 PASS";
        badge.className = "pass";
      } else if (msg.verify && msg.verify.passed === false) {
        badge.textContent = "对齐 FAIL";
        badge.className = "fail";
      } else if (!msg.acq_enabled) {
        badge.textContent = "无 EEG";
        badge.className = "na";
      } else {
        badge.textContent = "—";
        badge.className = "na";
      }
    }
    if (!msg.train_eligible) {
      el.summaryMsg.textContent += "（不可用于训练切窗）";
    }
    const aq = msg.acq_quality;
    if (aq && aq.drop_rate_pct != null) {
      el.summaryMsg.textContent += ` · 丢包率 ${aq.drop_rate_pct}%`;
    }
    const pipes = msg.phase4?.v2_pipes || msg.v2_acceptance?.phase4;
    if (pipes?.pipes && el.p4SummaryV2) {
      const parts = Object.entries(pipes.pipes).map(
        ([k, v]) => `${k}: ${v.n_windows ?? "—"} 窗`
      );
      el.p4SummaryV2.textContent =
        `v2 管道 · ${parts.join(" · ")}` +
        (pipes.merged_manifest ? ` · merged: ${pipes.merged_manifest}` : "");
      el.p4SummaryV2.classList.remove("hidden");
    }
    showPhase4Result(msg.phase4 || null);
    const btnP4 = document.getElementById("btn-phase4");
    if (btnP4) btnP4.disabled = !msg.acq_enabled;
  } else if (t === "phase4_ack") {
    const btnP4 = document.getElementById("btn-phase4");
    if (btnP4) {
      btnP4.disabled = false;
      btnP4.textContent = "一键 Phase4 切窗";
    }
    showPhase4Result(msg);
    if (msg.ok && msg.epochs_dir) {
      // 刷新文件列表提示指针已写入
      const li = document.createElement("li");
      li.className = "ok";
      li.textContent = `99_summary/phase4_pointer.json → ${msg.epochs_dir}`;
      el.summaryFiles.appendChild(li);
    }
  } else if (t === "questionnaire_ack") {
    const runEl = document.getElementById("q-status-run");
    const sumEl = document.getElementById("q-status");
    let text;
    if (msg.ok) {
      const s = msg.summary || {};
      text =
        `问卷已提交并保存 → ${msg.path || ""}` +
        (s.kinesthetic_mean != null ? ` · 动觉均分 ${s.kinesthetic_mean}` : "") +
        (s.involuntary ? ` · 不自主动手: ${s.involuntary}` : "") +
        (s.fatigue != null ? ` · 疲劳 ${s.fatigue}` : "");
    } else {
      text = `问卷未完成：${(msg.errors || []).join("；") || msg.message || "未知错误"}`;
    }
    if (runEl) runEl.textContent = text;
    if (sumEl) sumEl.textContent = text;
  } else if (t === "operator_hint") {
    const runEl = document.getElementById("q-status-run");
    if (runEl && msg.message) runEl.textContent = msg.message;
  } else if (t === "subject_page_opened") {
    if (!msg.ok) el.popupWarn.classList.remove("hidden");
  }
}

let subjectPageOpened = false;

function tryOpenSubject(force = false) {
  const cfgOpen = el.form.querySelector('[name="open_subject_page"]')?.checked !== false;
  if (!cfgOpen) return;
  if (subjectPageOpened && !force) return;
  const w = window.open(subjectUrl, "_blank");
  if (w) {
    subjectPageOpened = true;
  } else {
    el.popupWarn.classList.remove("hidden");
    send({ type: "open_subject_page" });
  }
}

function loadLocalDefaults() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveLocalDefaults(cfg) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
}

function startSession() {
  if (starting) return;
  subjectPageOpened = false;
  resetRunView();
  lockedConfig = null;
  sessionRoot = "";
  const cfg = formToRunConfig();
  hotkeysEnabled = cfg.ui.operator_hotkeys !== false;
  showErrors([]);
  if (cfg.ui.remember_last_config) saveLocalDefaults(cfg);
  send({ type: "session_start", run_config: cfg });
}

el.form.addEventListener("change", () => {
  syncAcqUi();
  syncPhaseModeUi();
  if (isV2Mode()) renderSetupTimelineV2();
  else renderSetupTimeline();
});
el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  startSession();
});

document.getElementById("btn-reset").addEventListener("click", () => {
  applyConfigToForm(builtinDefaults || defaultsFromServer);
  if (el.reuseBar) el.reuseBar.classList.add("hidden");
});

document.getElementById("btn-save-local").addEventListener("click", () => {
  const cfg = formToRunConfig();
  saveLocalDefaults(cfg);
  send({ type: "save_defaults", run_config: cfg });
});

document.getElementById("btn-export-cfg")?.addEventListener("click", () => {
  const cfg = formToRunConfig();
  const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `operator_config_${cfg.subject.subject_id}_${cfg.subject.session_id}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

document.getElementById("btn-import-cfg")?.addEventListener("change", async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const cfg = JSON.parse(text);
    applyConfigToForm(cfg);
    showErrors([]);
    alert("已导入配置（尚未开始；可再点保存为默认）");
  } catch (err) {
    showErrors([`导入失败: ${err}`]);
  }
  ev.target.value = "";
});

document.getElementById("btn-refresh-ports")?.addEventListener("click", () => {
  send({ type: "list_serial_ports" });
});

document.getElementById("btn-reuse-start")?.addEventListener("click", () => {
  startSession();
});
document.getElementById("btn-reuse-edit")?.addEventListener("click", () => {
  if (el.reuseBar) el.reuseBar.classList.add("hidden");
});

document.getElementById("btn-split").addEventListener("click", () => {
  send({ type: "operator", action: "split_session" });
});
document.getElementById("btn-questionnaire").addEventListener("click", () => {
  send({ type: "questionnaire_open" });
  const runEl = document.getElementById("q-status-run");
  if (runEl) runEl.textContent = "正在推送问卷到诱导页…（请保持诱导页打开）";
});
document.getElementById("btn-summary-questionnaire")?.addEventListener("click", () => {
  send({ type: "questionnaire_open" });
  const sumEl = document.getElementById("q-status");
  if (sumEl) sumEl.textContent = "正在推送问卷到诱导页…（请保持诱导页打开；提交后见 99_summary/）";
});
document.getElementById("btn-pause").addEventListener("click", () => {
  send({ type: "operator", action: paused ? "resume" : "pause" });
});
document.getElementById("btn-continue").addEventListener("click", () => {
  send({ type: "operator", action: "continue" });
});
document.getElementById("btn-gate").addEventListener("click", () => {
  send({ type: "operator", action: "gate_ok" });
});
el.btnV2Guidance?.addEventListener("click", () => {
  send({ type: "v2_guidance_confirm" });
  stopGuidanceCountdown();
  if (el.btnV2Guidance) el.btnV2Guidance.disabled = true;
  if (el.v2StageHint) el.v2StageHint.textContent = "已确认引导，会话继续…";
});
el.btnV3Guidance?.addEventListener("click", () => {
  send({ type: "v2_guidance_confirm" });
  stopGuidanceCountdown();
  if (el.btnV3Guidance) el.btnV3Guidance.disabled = true;
});
document.getElementById("btn-reject").addEventListener("click", () => {
  send({ type: "operator", action: "reject" });
});
document.getElementById("btn-reopen").addEventListener("click", () => {
  tryOpenSubject(true);
  send({ type: "open_subject_page" });
});
el.btnRestart?.addEventListener("click", () => {
  if (
    !confirm(
      "确认重新开始？当前场次将中止并返回配置页（已采集数据会保留，可立即再开一场）。"
    )
  ) {
    return;
  }
  restartAfterAbort = true;
  showRunAlert("正在中止，随后返回配置页…", "abort");
  send({ type: "operator", action: "abort" });
});
el.btnV4ToV3?.addEventListener("click", () => {
  const v3Radio = el.form.querySelector('input[name="phase_mode"][value="v3_session"]');
  if (v3Radio) v3Radio.checked = true;
  syncPhaseModeUi();
  restartAfterAbort = true;
  showRunAlert("v4 已达标，正在结束并返回配置页…", "ok");
  send({ type: "operator", action: "abort" });
});
document.getElementById("btn-abort").addEventListener("click", () => {
  if (confirm("确认中止本场实验？已写入数据将尽量保留。")) {
    showRunAlert("正在中止会话…", "abort");
    send({ type: "operator", action: "abort" });
  }
});

document.getElementById("btn-open-folder").addEventListener("click", () => {
  if (sessionRoot) send({ type: "open_folder", path: sessionRoot });
});
document.getElementById("btn-phase4")?.addEventListener("click", () => {
  if (!sessionRoot) {
    alert("无会话目录");
    return;
  }
  const btn = document.getElementById("btn-phase4");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "切窗中…";
  }
  if (el.phase4Msg) el.phase4Msg.textContent = "Phase4 切窗进行中（仅 acquire + 未 reject）…";
  const runMode = document.getElementById("p4-run-mode")?.value || "fixed";
  const runWin = Number(document.getElementById("p4-run-win")?.value) || 2;
  const runHop = Number(document.getElementById("p4-run-hop")?.value) || 100;
  send({
    type: "run_phase4",
    path: sessionRoot,
    phase4: { window_mode: runMode, win_sec: runWin, hop_ms: runHop },
  });
});
document.getElementById("btn-copy-path").addEventListener("click", async () => {
  if (!sessionRoot) return;
  try {
    await navigator.clipboard.writeText(sessionRoot);
    alert("已复制路径");
  } catch {
    prompt("复制路径：", sessionRoot);
  }
});
document.getElementById("btn-again").addEventListener("click", () => {
  starting = false;
  resetRunView();
  lockedConfig = null;
  sessionRoot = "";
  showView("setup");
  maybeShowReuseBar(defaultsFromServer || loadLocalDefaults());
});

window.addEventListener("keydown", (e) => {
  if (el.run.classList.contains("hidden")) return;
  if (!hotkeysEnabled) return;
  if (e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === "b") send({ type: "operator", action: "split_session" });
  if (k === "q") send({ type: "questionnaire_open" });
  if (k === "p") send({ type: "operator", action: "toggle_pause" });
  if (k === "n") send({ type: "operator", action: "continue" });
  if (k === "g") send({ type: "operator", action: "gate_ok" });
  if (k === "r") send({ type: "operator", action: "reject" });
  if (e.key === "Escape") {
    if (confirm("确认中止？")) send({ type: "operator", action: "abort" });
  }
});

const hash = (location.hash || "#setup").replace("#", "");
showView(["setup", "run", "summary"].includes(hash) ? hash : "setup");
syncAcqUi();
syncPhaseModeUi();
renderSetupTimeline();
connect();
