const WS_URL =
  new URLSearchParams(location.search).get("ws") ||
  `ws://${location.hostname || "127.0.0.1"}:8765`;

const STORAGE_KEY = "experiment_game_operator_defaults_v1";
const SUBJECT_LOGIN_KEY = "experiment_game_subject_login_v1";

const el = {
  login: document.getElementById("view-login"),
  loginForm: document.getElementById("login-form"),
  loginError: document.getElementById("login-error"),
  loginLast: document.getElementById("login-last"),
  subjectBar: document.getElementById("subject-bar"),
  subjectBarId: document.getElementById("subject-bar-id"),
  subjectBarSession: document.getElementById("subject-bar-session"),
  subjectBarWeights: document.getElementById("subject-bar-weights"),
  setupSubjectId: document.getElementById("setup-subject-id"),
  setupSessionId: document.getElementById("setup-session-id"),
  sessionIdList: document.getElementById("session-id-list"),
  sessionIdSuggestBtn: document.getElementById("session-id-suggest-btn"),
  sessionIdBoardMap: document.getElementById("session-id-board-map"),
  sessionIdHint: document.getElementById("session-id-hint"),
  sessionOverwriteWrap: document.getElementById("session-overwrite-wrap"),
  sessionOverwrite: document.getElementById("session-overwrite"),
  ftPanel: document.getElementById("ft-panel"),
  ftCurrentWeights: document.getElementById("ft-current-weights"),
  ftSessionList: document.getElementById("ft-session-list"),
  ftExcludeInvalid: document.getElementById("ft-exclude-invalid"),
  ftUseReplay: document.getElementById("ft-use-replay"),
  ftLeaveNext: document.getElementById("ft-leave-next"),
  ftRampHint: document.getElementById("ft-ramp-hint"),
  ftReplayRatio: document.getElementById("ft-replay-ratio"),
  ftReplayHint: document.getElementById("ft-replay-hint"),
  btnFtReplayAdopt: document.getElementById("btn-ft-replay-adopt"),
  ftEarlyStop: document.getElementById("ft-early-stop"),
  ftMaxEpochs: document.getElementById("ft-max-epochs"),
  ftPatience: document.getElementById("ft-patience"),
  ftFixedEpochs: document.getElementById("ft-fixed-epochs"),
  ftDeterministic: document.getElementById("ft-deterministic"),
  ftSeed: document.getElementById("ft-seed"),
  setupFtReplayHint: document.getElementById("setup-ft-replay-hint"),
  btnSetupReplayAdopt: document.getElementById("btn-setup-replay-adopt"),
  ftResult: document.getElementById("ft-result"),
  ftStatus: document.getElementById("ft-status"),
  btnFtStart: document.getElementById("btn-ft-start"),
  btnFtPromote: document.getElementById("btn-ft-promote"),
  btnFtKeep: document.getElementById("btn-ft-keep"),
  btnNextSession: document.getElementById("btn-next-session"),
  btnSessionNoRecord: document.getElementById("btn-session-no-record"),
  simRunQueue: document.getElementById("sim-run-queue"),
  simCampaignSelect: document.getElementById("sim-campaign-select"),
  simCampaignStatus: document.getElementById("sim-campaign-status"),
  btnSimCampaignCreate: document.getElementById("btn-sim-campaign-create"),
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
  summaryWindowAcc: document.getElementById("summary-window-acc"),
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
  setupTimelineV3: document.getElementById("setup-timeline-v3"),
  timingHintV3: document.getElementById("timing-hint-v3"),
  phaseStepsV2: document.getElementById("phase-steps-v2"),
  stV2Round: document.getElementById("st-v2-round"),
  stV2Score: document.getElementById("st-v2-score"),
  stSessionScoreWrap: document.getElementById("st-session-score-wrap"),
  stSessionScore: document.getElementById("st-session-score"),
  stWeights: document.getElementById("st-weights"),
  modelPreset: document.getElementById("model-preset"),
  modelCurrentName: document.getElementById("model-current-name"),
  modelCurrentMode: document.getElementById("model-current-mode"),
  s3TaskCkpt: document.getElementById("s3_task_ckpt"),
  s3ThreeCkpt: document.getElementById("s3_three_ckpt"),
  modelWeightHint: document.getElementById("model-weight-hint"),
  v2CalProg: document.getElementById("v2-cal-prog"),
  v2GameProg: document.getElementById("v2-game-prog"),
  v2Subblock: document.getElementById("v2-subblock"),
  v2FtStatus: document.getElementById("v2-ft-status"),
  v2ScoreNum: document.getElementById("v2-score-num"),
  v2ScoreFill: document.getElementById("v2-score-fill"),
  v2SessionScoreNum: document.getElementById("v2-session-score-num"),
  v2SessionScoreFill: document.getElementById("v2-session-score-fill"),
  v2SessionScoreHint: document.getElementById("v2-session-score-hint"),
  v2SessionScoreRules: document.getElementById("v2-session-score-rules"),
  v2SessionScoreBy: document.getElementById("v2-session-score-by"),
  v2WeakMi: document.getElementById("v2-weak-mi"),
  v2AcceptDetail: document.getElementById("v2-accept-detail"),
  p4SummaryV2: document.getElementById("p4-summary-v2"),
  runAlert: document.getElementById("run-alert"),
  v2GuidanceCountdown: document.getElementById("v2-guidance-countdown"),
  v3Panel: document.getElementById("v3-panel"),
  v3BlockProg: document.getElementById("v3-block-prog"),
  v3TrialProg: document.getElementById("v3-trial-prog"),
  v3Cond: document.getElementById("v3-cond"),
  v3SessionScoreNum: document.getElementById("v3-session-score-num"),
  v3SessionScoreFill: document.getElementById("v3-session-score-fill"),
  v3SessionScoreHint: document.getElementById("v3-session-score-hint"),
  v3SessionScoreRules: document.getElementById("v3-session-score-rules"),
  v3SessionScoreBy: document.getElementById("v3-session-score-by"),
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
const LIVE_LABEL_NAMES = { 0: "Rest", 1: "Left", 2: "Right" };
const LIVE_TRIAL_VOTES = { v2: { 0: 0, 1: 0, 2: 0 }, v3: { 0: 0, 1: 0, 2: 0 } };
const SESSION_TRIAL_VOTES = {
  v2: { history: [], view: "live" },
  v3: { history: [], view: "live" },
};

function _voteCountFromMap(vc, cls) {
  if (!vc || typeof vc !== "object") return 0;
  const n = vc[cls] ?? vc[String(cls)];
  return Number.isFinite(Number(n)) ? Number(n) : 0;
}

function _labelTag(label, name) {
  if (label == null) return name || "—";
  const nm = name || LIVE_LABEL_NAMES[label] || String(label);
  return `${nm} (${label})`;
}

function resetSessionVoteHistory() {
  for (const prefix of ["v2", "v3"]) {
    SESSION_TRIAL_VOTES[prefix] = { history: [], view: "live" };
    LIVE_TRIAL_VOTES[prefix] = { 0: 0, 1: 0, 2: 0 };
    renderVotePanel(prefix);
  }
}

function recordTrialVoteHistory(prefix, ctx, summary) {
  if (!summary?.vote_counts) return;
  const vc = summary.vote_counts;
  const state = SESSION_TRIAL_VOTES[prefix];
  const label = summary.label ?? ctx?.label;
  const pred = summary.pred;
  const rec = {
    trial_id: ctx?.trial_id ?? state.history.length + 1,
    label,
    label_name: ctx?.label_name || LIVE_LABEL_NAMES[label] || String(label ?? "—"),
    pred,
    pred_name: pred != null ? LIVE_LABEL_NAMES[pred] ?? String(pred) : "—",
    correct: summary.correct,
    vote_counts: {
      0: _voteCountFromMap(vc, 0),
      1: _voteCountFromMap(vc, 1),
      2: _voteCountFromMap(vc, 2),
    },
  };
  const dup = state.history.findIndex((h) => h.trial_id === rec.trial_id);
  if (dup >= 0) state.history[dup] = rec;
  else state.history.push(rec);
  state.view = "live";
  applyTrialVotesFromSummary(prefix, summary);
  renderVotePanel(prefix);
}

function navigateVoteHistory(prefix, delta) {
  const state = SESSION_TRIAL_VOTES[prefix];
  const n = state.history.length;
  if (n === 0) return;
  if (state.view === "live") {
    if (delta < 0) state.view = n - 1;
    else return;
  } else {
    const next = state.view + delta;
    if (next < 0) return;
    if (next >= n) state.view = "live";
    else state.view = next;
  }
  renderVotePanel(prefix);
}

function renderVotePanel(prefix) {
  const state = SESSION_TRIAL_VOTES[prefix];
  const hist = state.history;
  const viewing = state.view !== "live" ? hist[state.view] : null;
  const votes = viewing ? viewing.vote_counts : LIVE_TRIAL_VOTES[prefix] || { 0: 0, 1: 0, 2: 0 };
  let total = 0;
  for (let i = 0; i < 3; i++) {
    const n = votes[i] || 0;
    total += n;
    const cell = document.getElementById(`${prefix}-v${i}-votes`);
    if (cell) cell.textContent = String(n);
  }
  const head = document.getElementById(`${prefix}-votes-total`);
  if (head) head.textContent = `共 ${total} 窗`;

  const navLabel = document.getElementById(`${prefix}-votes-nav-label`);
  const prevBtn = document.getElementById(`${prefix}-votes-prev`);
  const nextBtn = document.getElementById(`${prefix}-votes-next`);
  const meta = document.getElementById(`${prefix}-votes-meta`);
  const box = document.querySelector(`#${prefix}-live-judge .live-judge-votes`);

  if (state.view === "live") {
    if (navLabel) navLabel.textContent = hist.length ? `当前试次 · 已完成 ${hist.length}` : "当前试次";
    if (prevBtn) prevBtn.disabled = hist.length === 0;
    if (nextBtn) nextBtn.disabled = true;
    if (meta) meta.innerHTML = "";
    box?.classList.remove("is-history");
  } else if (viewing) {
    const pos = Number(state.view) + 1;
    if (navLabel) navLabel.textContent = `回看 ${pos}/${hist.length}`;
    if (prevBtn) prevBtn.disabled = state.view <= 0;
    if (nextBtn) nextBtn.disabled = false;
    const ok = viewing.correct === true;
    const fail = viewing.correct === false;
    const mark = ok ? '<span class="vote-meta-correct">✓</span>' : fail ? '<span class="vote-meta-wrong">✗</span>' : "";
    if (meta) {
      meta.innerHTML =
        `试次 ${viewing.trial_id} ${mark}<br>` +
        `正式 ${_labelTag(viewing.label, viewing.label_name)} · ` +
        `预测 ${_labelTag(viewing.pred, viewing.pred_name)}`;
    }
    box?.classList.add("is-history");
  }
}

function resetTrialVotes(prefix) {
  LIVE_TRIAL_VOTES[prefix] = { 0: 0, 1: 0, 2: 0 };
  SESSION_TRIAL_VOTES[prefix].view = "live";
  renderVotePanel(prefix);
}

function renderTrialVotes(prefix) {
  if (SESSION_TRIAL_VOTES[prefix]?.view !== "live") return;
  renderVotePanel(prefix);
}

function applyTrialVotesFromSummary(prefix, summary) {
  if (!summary?.vote_counts) return;
  const vc = summary.vote_counts;
  LIVE_TRIAL_VOTES[prefix] = {
    0: _voteCountFromMap(vc, 0),
    1: _voteCountFromMap(vc, 1),
    2: _voteCountFromMap(vc, 2),
  };
  if (SESSION_TRIAL_VOTES[prefix].view === "live") renderVotePanel(prefix);
}

function bindVoteHistorySwipe(prefix) {
  const box = document.querySelector(`#${prefix}-live-judge .live-judge-votes`);
  if (!box || box.dataset.swipeBound) return;
  box.dataset.swipeBound = "1";
  let x0 = null;
  box.addEventListener(
    "touchstart",
    (e) => {
      x0 = e.changedTouches[0]?.clientX ?? null;
    },
    { passive: true },
  );
  box.addEventListener(
    "touchend",
    (e) => {
      if (x0 == null) return;
      const x1 = e.changedTouches[0]?.clientX ?? x0;
      const dx = x1 - x0;
      x0 = null;
      if (Math.abs(dx) < 40) return;
      navigateVoteHistory(prefix, dx < 0 ? 1 : -1);
    },
    { passive: true },
  );
}

function _liveJudgeEls(prefix) {
  return {
    label: document.getElementById(`${prefix}-live-label`),
    pred: document.getElementById(`${prefix}-live-pred`),
    correct: document.getElementById(`${prefix}-live-correct`),
    trel: document.getElementById(`${prefix}-live-trel`),
    fills: [0, 1, 2].map((i) => document.getElementById(`${prefix}-p${i}-fill`)),
    pcts: [0, 1, 2].map((i) => document.getElementById(`${prefix}-p${i}-pct`)),
    rows: document.querySelectorAll(`#${prefix}-live-judge .live-prob-row`),
  };
}

function resetLiveJudge(prefix) {
  const box = _liveJudgeEls(prefix);
  if (box.label) box.label.textContent = "—";
  if (box.pred) box.pred.textContent = "—";
  if (box.correct) {
    box.correct.textContent = "";
    box.correct.className = "live-judge-correct";
  }
  if (box.trel) box.trel.textContent = "";
  box.fills.forEach((f) => {
    if (f) f.style.width = "0%";
  });
  box.pcts.forEach((p) => {
    if (p) p.textContent = "—";
  });
  box.rows.forEach((r) => r.classList.remove("is-pred"));
  resetTrialVotes(prefix);
}

function updateLiveJudge(prefix, msg) {
  const box = _liveJudgeEls(prefix);
  if (!box.label) return;
  const ctx = msg.ctx || {};
  const data = msg.data || {};
  const stage = msg.stage;

  if (stage === "trial_start" || stage === "iti") {
    if (stage === "trial_start") resetLiveJudge(prefix);
  }

  if (stage === "trial_end") {
    const sum = data.summary;
    if (sum?.vote_counts) recordTrialVoteHistory(prefix, ctx, sum);
  }

  const lab = ctx.label;
  const labName =
    ctx.label_name ||
    (lab != null ? LIVE_LABEL_NAMES[lab] ?? String(lab) : null);
  if (labName != null) {
    box.label.textContent =
      lab != null ? `${labName} (${lab})` : labName;
  }
  if (msg.ctx?.label != null && el.stLabel) {
    const zh = V3_LABEL_NAMES[lab] || labName;
    el.stLabel.textContent = `${zh} (${lab})`;
  }

  if (stage !== "judge") return;

  if (data.signal_bad) {
    if (box.pred) box.pred.textContent = "信号差";
    if (box.correct) {
      box.correct.textContent = "";
      box.correct.className = "live-judge-correct";
    }
    if (box.trel && data.t_rel != null) {
      box.trel.textContent = `t=${Number(data.t_rel).toFixed(1)}s`;
    }
    box.fills.forEach((f) => {
      if (f) f.style.width = "0%";
    });
    box.pcts.forEach((p) => {
      if (p) p.textContent = "—";
    });
    box.rows.forEach((r) => r.classList.remove("is-pred"));
    return;
  }

  const votePred =
    data.gated_pred != null && data.gated_pred !== ""
      ? Number(data.gated_pred)
      : data.pred != null
        ? Number(data.pred)
        : NaN;
  if (Number.isFinite(votePred) && votePred >= 0 && votePred <= 2) {
    const bucket = LIVE_TRIAL_VOTES[prefix] || { 0: 0, 1: 0, 2: 0 };
    bucket[votePred] = (bucket[votePred] || 0) + 1;
    LIVE_TRIAL_VOTES[prefix] = bucket;
    renderTrialVotes(prefix);
  }

  const pred = data.pred;
  const predName =
    data.pred_name ||
    (pred != null ? LIVE_LABEL_NAMES[pred] ?? String(pred) : null);
  if (box.pred) {
    box.pred.textContent =
      predName != null
        ? pred != null
          ? `${predName} (${pred})`
          : predName
        : "—";
  }
  if (box.correct) {
    if (data.correct === true) {
      box.correct.textContent = "✓ 正确";
      box.correct.className = "live-judge-correct chk-ok";
    } else if (data.correct === false) {
      box.correct.textContent = "✗ 错误";
      box.correct.className = "live-judge-correct chk-fail";
    } else {
      box.correct.textContent = "";
      box.correct.className = "live-judge-correct";
    }
  }
  if (box.trel) {
    box.trel.textContent =
      data.t_rel != null ? `t=${Number(data.t_rel).toFixed(1)}s` : "";
  }

  const pThree = Array.isArray(data.p_three) ? data.p_three : null;
  for (let i = 0; i < 3; i++) {
    const p = pThree != null ? Number(pThree[i]) : NaN;
    const pct = Number.isFinite(p) ? Math.max(0, Math.min(100, p * 100)) : null;
    if (box.fills[i]) box.fills[i].style.width = pct != null ? `${pct}%` : "0%";
    if (box.pcts[i]) {
      box.pcts[i].textContent = pct != null ? `${pct.toFixed(0)}%` : "—";
    }
    if (box.rows[i]) {
      box.rows[i].classList.toggle("is-pred", pred != null && Number(pred) === i);
    }
  }
}

function updateLiveJudgePanels(msg) {
  const v3Active = !el.v3Panel?.classList.contains("hidden");
  const v2Active = !el.v2Panel?.classList.contains("hidden");
  if (v2Active) updateLiveJudge("v2", msg);
  if (v3Active) updateLiveJudge("v3", msg);
}
const V3_MU_ERD_OK = -15;
const V3_LAT_OK = 8;
const V3_CH = ["C3", "C4", "CZ", "CP3", "CP4", "CPZ", "FC3", "FC4"];
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

/** 本场累计得分：MI 试次 +1，Cue前静息 +0.5；满分 = n_mi + n_mi×0.5（36→54） */
const SESSION_SCORE = {
  score: 0,
  max: null,
  done: 0,
  nMi: null,
  by: { left: 0, right: 0, pre_cue_rest: 0 },
};

function formatScoreHalf(v) {
  const n = Number(v) || 0;
  return Math.abs(n - Math.round(n)) < 1e-9 ? String(Math.round(n)) : n.toFixed(1);
}

function formatWindowAccPct(acc) {
  if (acc == null || !Number.isFinite(Number(acc))) return null;
  return `${(Number(acc) * 100).toFixed(0)}%`;
}

/** 结束页：本场得分 + 窗级识别率（非试次多数票） */
function appendSessionScoreSummary(elMsg, vs) {
  if (!elMsg || !vs) return;
  if (vs.session_score != null && vs.session_score_max != null) {
    elMsg.textContent +=
      ` · 本场得分 ${formatScoreHalf(vs.session_score)}/${formatScoreHalf(vs.session_score_max)}`;
  }
  renderSummaryWindowAcc(vs);
}

function renderSummaryWindowAcc(vs) {
  const node = el.summaryWindowAcc;
  if (!node) return;
  if (!vs) {
    node.classList.add("hidden");
    node.innerHTML = "窗级识别率 <strong>—</strong>";
    return;
  }
  const wa = vs.window_acc ?? vs.report?.overall?.acc_window;
  const waTxt = formatWindowAccPct(wa);
  if (!waTxt) {
    node.classList.add("hidden");
    node.innerHTML = "窗级识别率 <strong>—</strong>";
    return;
  }
  const n = vs.window_acc_n ?? vs.report?.overall?.n_windows;
  const nTxt = n != null ? `（${n} 窗 · L/R）` : "（L/R 逐窗）";
  node.classList.remove("hidden");
  node.innerHTML = `模型窗级识别率 <strong>${waTxt}</strong> <span class="hint">${nTxt}</span>`;
}

/** OpenBMI：n MI 试次 → 满分 n + n×0.5（含 Cue前静息） */
function openbmiSessionScoreMax(nMi, interTrialRestS = 4) {
  const n = Math.max(0, Number(nMi) || 0);
  if (!(Number(interTrialRestS) > 1e-6)) return n;
  return n + n * 0.5;
}

function sessionScoreByMax(nMi, interTrialRestS = 4) {
  const n = Math.max(0, Math.round(Number(nMi) || 0));
  const nL = Math.floor(n / 2);
  const nR = n - nL;
  const restMax = Number(interTrialRestS) > 1e-6 ? n * 0.5 : 0;
  return { left: nL, right: nR, pre_cue_rest: restMax };
}

function scoreByText(by, nMi, interTrialRestS = 4) {
  const caps = sessionScoreByMax(nMi, interTrialRestS);
  const b = by || {};
  const left = formatScoreHalf(b.left || 0);
  const right = formatScoreHalf(b.right || 0);
  const cueRest = formatScoreHalf(b.pre_cue_rest || 0);
  const leftMax = caps.left > 0 ? formatScoreHalf(caps.left) : "—";
  const rightMax = caps.right > 0 ? formatScoreHalf(caps.right) : "—";
  const cueMax = caps.pre_cue_rest > 0 ? formatScoreHalf(caps.pre_cue_rest) : "—";
  return (
    `Left <strong>${left}/${leftMax}</strong> · ` +
    `Right <strong>${right}/${rightMax}</strong> · ` +
    `Cue前静息 <strong>${cueRest}/${cueMax}</strong>`
  );
}

function scoreRulesText(nMi, interTrialRestS = 4) {
  const n = Math.max(0, Math.round(Number(nMi) || 0));
  if (!(n > 0)) {
    return "计分：MI Left/Right 判对 +1；Cue前静息（正式 Rest）判对 +0.5";
  }
  const nL = Math.floor(n / 2);
  const nR = n - nL;
  if (!(Number(interTrialRestS) > 1e-6)) {
    return `满分 ${n}：Left ${nL}×1 + Right ${nR}×1（无 Cue前静息计分）`;
  }
  const restMax = n * 0.5;
  const total = n + restMax;
  return (
    `满分 ${formatScoreHalf(total)}：` +
    `Left ${nL} 试 ×1（满分 ${nL}）· ` +
    `Right ${nR} 试 ×1（满分 ${nR}）· ` +
    `Cue前静息 ${n} 段 ×0.5（满分 ${formatScoreHalf(restMax)}）`
  );
}

function resetSessionScore(max, opts = {}) {
  SESSION_SCORE.score = 0;
  SESSION_SCORE.max = max != null && Number.isFinite(Number(max)) ? Number(max) : null;
  SESSION_SCORE.done = 0;
  SESSION_SCORE.by = { left: 0, right: 0, pre_cue_rest: 0 };
  if (opts.nMi != null) SESSION_SCORE.nMi = Number(opts.nMi);
  else if (SESSION_SCORE.max != null && opts.inferNMi !== false) {
    // 54 → 36；无静息时 max=nMi
    const m = SESSION_SCORE.max;
    SESSION_SCORE.nMi = Math.abs(m - Math.round(m / 1.5) * 1.5) < 1e-6
      ? Math.round(m / 1.5)
      : Math.round(m);
  }
  if (opts.interTrialRestS != null) SESSION_SCORE.restS = Number(opts.interTrialRestS);
  renderSessionScore();
}

function normalizeScoreBy(bySrc) {
  if (!bySrc || typeof bySrc !== "object") return null;
  return {
    left: Number(bySrc.left) || 0,
    right: Number(bySrc.right) || 0,
    pre_cue_rest: Number(bySrc.pre_cue_rest) || 0,
  };
}

/** 后端未带 session_score_by 时，按本条 stage 本地累加（防旧进程/漏字段） */
function bumpScoreByFromStage(msg) {
  const stage = msg?.stage;
  const summary = msg?.data?.summary;
  if (!summary || summary.score == null) return false;
  const pts = Number(summary.score) || 0;
  if (!(pts > 0)) return false;
  const lab = msg?.ctx?.label;
  let key = null;
  if (stage === "pre_cue_rest_end") key = "pre_cue_rest";
  else if (stage === "trial_end") {
    // 正式 MI 仅 Left/Right；Rest(MI) 不计分
    if (lab === 1 || lab === "1") key = "left";
    else if (lab === 2 || lab === "2") key = "right";
  }
  if (!key) return false;
  SESSION_SCORE.by[key] = (Number(SESSION_SCORE.by[key]) || 0) + pts;
  return true;
}

function applySessionScore(msg) {
  if (!msg) return;
  const prog = msg.progress && typeof msg.progress === "object" ? msg.progress : null;
  const src = prog || msg;
  if (src.session_score_max != null) SESSION_SCORE.max = Number(src.session_score_max);
  if (msg.session_score_max != null) SESSION_SCORE.max = Number(msg.session_score_max);
  if (src.session_score != null) SESSION_SCORE.score = Number(src.session_score);
  if (msg.session_score != null) SESSION_SCORE.score = Number(msg.session_score);
  if (src.session_trials_done != null) SESSION_SCORE.done = Number(src.session_trials_done);
  if (msg.session_trials_done != null) SESSION_SCORE.done = Number(msg.session_trials_done);
  const bySrc = normalizeScoreBy(src.session_score_by || msg.session_score_by);
  if (bySrc) {
    SESSION_SCORE.by = bySrc;
  } else {
    bumpScoreByFromStage(msg);
  }
  const blocks = src.blocks_total ?? msg.v3_config_effective?.blocks;
  const tpb = src.trials_per_block ?? msg.v3_config_effective?.trials_per_block;
  if (blocks != null && tpb != null) SESSION_SCORE.nMi = Number(blocks) * Number(tpb);
  renderSessionScore();
}

function renderSessionScore() {
  const { score, max, done, nMi, by } = SESSION_SCORE;
  const restS = SESSION_SCORE.restS != null ? SESSION_SCORE.restS : 4;
  const maxTxt = max != null ? formatScoreHalf(max) : "—";
  const scoreTxt = `${formatScoreHalf(score)}/${maxTxt}`;
  if (el.stSessionScore) el.stSessionScore.textContent = scoreTxt;
  if (el.v2SessionScoreNum) el.v2SessionScoreNum.textContent = scoreTxt;
  if (el.v3SessionScoreNum) el.v3SessionScoreNum.textContent = scoreTxt;
  const pct = max > 0 ? (Number(score) / max) * 100 : 0;
  const w = `${Math.min(100, Math.max(0, pct))}%`;
  if (el.v2SessionScoreFill) el.v2SessionScoreFill.style.width = w;
  if (el.v3SessionScoreFill) el.v3SessionScoreFill.style.width = w;
  const nForBy = nMi != null ? nMi : max != null ? Math.round(Number(max) / 1.5) : 0;
  const byHtml = scoreByText(by, nForBy, restS);
  if (el.v2SessionScoreBy) el.v2SessionScoreBy.innerHTML = byHtml;
  if (el.v3SessionScoreBy) el.v3SessionScoreBy.innerHTML = byHtml;
  const rules = scoreRulesText(nForBy, restS);
  if (el.v2SessionScoreRules) el.v2SessionScoreRules.textContent = rules;
  if (el.v3SessionScoreRules) el.v3SessionScoreRules.textContent = rules;
  const hint =
    max != null
      ? `已完成 ${done} / ${nMi != null ? nMi : "—"} MI 试次（静息另计）`
      : "";
  if (el.v2SessionScoreHint) el.v2SessionScoreHint.textContent = hint;
  if (el.v3SessionScoreHint) el.v3SessionScoreHint.textContent = hint;
}

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

function v3BaselineRestS() {
  return Number(el.form?.elements?.namedItem("v3_baseline_rest_s")?.value) || 30;
}

function v3BaselineWaitHtml() {
  const s = v3BaselineRestS();
  return `<div class="v3-pbar-wait">基线采集中…（块前 ${s}s 静息结束后显示 ERD%）</div>`;
}

function renderV3PowerBars(msg) {
  if (!el.v3PowerBars) return;
  const sigOk = msg.signal_ok !== false;
  if (!V3_EEG_STATE.baselineReady) {
    V3_POWER_ROWS = null;
    V3_POWER_NOTE = null;
    const note = sigOk ? "" : `<div class="v3-pbar-signal">${v3SignalNoteText(msg)}，基线可能无效</div>`;
    el.v3PowerBars.innerHTML = note + v3BaselineWaitHtml();
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
      `<div class="v3-erd-grid-label">各通道 mu ERD%（vs 开场 30s seed / 试次间 Rest 基线）</div>`,
      `<div class="v3-erd-grid">${chCells}</div>`,
      `<div class="v3-lat-row">偏侧 ${lat != null ? lat.toFixed(1) : "—"}pp <span class="v3-thr">（≥${V3_LAT_OK}pp）</span> <span class="${latCls}">${lat != null && lat >= V3_LAT_OK ? "✓" : ""}</span></div>`,
      `<div class="v3-block-grade"><span class="v3-block-label">块累计</span> ${bg.grade || "—"} · n=${f.block_n_mi_trials ?? "—"} MI / ${f.block_n_rest_segments ?? f.block_n_rest_trials ?? "—"} Rest段</div>`,
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
  applySessionScore(msg);
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
  if (msg.sim_skipped) {
    // 等首次试次间 Rest 再 ready；此时只提示
    V3_EEG_STATE.baselineReady = false;
    V3_EEG_STATE.baselineMu = [];
    V3_EEG_STATE.baselineBeta = [];
    V3_POWER_ROWS = null;
    if (el.v3PowerBars) {
      el.v3PowerBars.innerHTML =
        `<div class="v3-pbar-wait">${msg.message || "仿真：等待试次间 Rest 建立 ERD 基线…"}</div>`;
    }
    return;
  }
  V3_EEG_STATE.baselineReady = true;
  V3_EEG_STATE.baselineMu = msg.baseline_mu || [];
  V3_EEG_STATE.baselineBeta = msg.baseline_beta || [];
  V3_POWER_ROWS = null;
  if (msg.sim_rest_seed && el.v3PowerBars && !(msg.baseline_mu || []).length) {
    el.v3PowerBars.innerHTML =
      `<div class="v3-pbar-wait">${msg.message || "仿真：ERD 基线已更新"}</div>`;
  }
  if (msg.hat_check || msg.hat_verdict) showV3HatCheck(msg);
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
  ["v3_inter_trial_rest_s", "inter_trial_rest_s", "float"],
];

/** @type {Array<{id:string,label:string,weight_label?:string,model_name?:string,task:string,three:string,ok?:boolean}>} */
let MODEL_PRESETS = [];
/** @type {string} */
let ACTIVE_MODEL_LABEL = "";
/** @type {Record<string, string>} Campaign R0 选定的零样本模型 preset id */
let campaignModelPicks = {};

const ZERO_SAMPLE_PRESET_IDS = new Set(["openbmi_baseline", "e1f_four_member"]);

function isZeroSamplePresetId(id) {
  return ZERO_SAMPLE_PRESET_IDS.has(String(id || ""));
}

function isZeroSamplePreset(p) {
  return isZeroSamplePresetId(p?.id);
}

function presetWeightLabel(p) {
  return p?.weight_label || p?.label || p?.id || "—";
}

function presetModelName(p) {
  return p?.model_name || p?.id || "—";
}

function campaignRemainingRuns(c) {
  const consumed = new Set(c?.runs_consumed || []);
  return (c?.session_queue || []).filter((r) => !consumed.has(r));
}

function inferCampaignLockedModelPresetId(campaign) {
  if (!campaign) return null;
  if (campaign.locked_model_preset_id) return campaign.locked_model_preset_id;
  const consumed = campaign.runs_consumed || [];
  if (!consumed.length) return null;
  const ft = findLatestPresetForCampaign(campaign.campaign_id);
  if (ft) {
    return ft.readout_mode === "e1f" ? "e1f_four_member" : "openbmi_baseline";
  }
  const pick = campaignModelPicks[campaign.campaign_id];
  if (pick && isZeroSamplePresetId(pick)) return pick;
  const e1f = MODEL_PRESETS.find((p) => p.id === "e1f_four_member" && p.ok !== false);
  return e1f ? "e1f_four_member" : "openbmi_baseline";
}

function presetAllowedForCampaign(p, campaign) {
  if (!campaign || !activeSimMode) return true;
  const cid = campaign.campaign_id;
  const consumed = campaign.runs_consumed || [];
  const lockedModel = inferCampaignLockedModelPresetId(campaign);
  const r0Pick = campaignModelPicks[cid];

  if (isZeroSamplePreset(p)) {
    if (consumed.length) return p.id === lockedModel;
    if (r0Pick) return p.id === r0Pick;
    return true;
  }
  if (p.kind === "current") return false;
  if (p.campaign_id) return p.campaign_id === cid;
  return false;
}

function visiblePresetsForCampaign() {
  if (!activeCampaign || !activeSimMode) return MODEL_PRESETS.slice();
  return MODEL_PRESETS.filter((p) => presetAllowedForCampaign(p, activeCampaign));
}

function syncCampaignWeightLockUi() {
  const campaignOn = Boolean(activeCampaign && activeSimMode);
  const consumed = activeCampaign?.runs_consumed || [];
  const lockedModel = inferCampaignLockedModelPresetId(activeCampaign);
  if (el.s3TaskCkpt) el.s3TaskCkpt.readOnly = campaignOn;
  if (el.s3ThreeCkpt) el.s3ThreeCkpt.readOnly = campaignOn;
  if (el.modelWeightHint) {
    if (!campaignOn) {
      el.modelWeightHint.textContent =
        "路径相对仓库根目录；v2/v3 共用。保存默认配置后下次自动带入。";
    } else if (consumed.length && lockedModel) {
      const mp = MODEL_PRESETS.find((p) => p.id === lockedModel);
      el.modelWeightHint.textContent =
        `Campaign 已锁定模型「${presetModelName(mp)}」；仅可选用本周期权重，不可切换模型或其他 Campaign 权重。`;
    } else {
      el.modelWeightHint.textContent =
        "Campaign R0：请选择零样本模型（E1f / OpenBMI）；开跑后将锁定，本周期不可更换。";
    }
  }
  refreshModelPresetSelectOptions();
}

function refreshModelPresetSelectOptions() {
  const sel = el.modelPreset;
  if (!sel) return;
  const cur = sel.value;
  const visible = visiblePresetsForCampaign();
  const visibleIds = new Set(visible.map((p) => p.id));
  sel.innerHTML = "";
  for (const p of visible) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.ok === false ? `${presetWeightLabel(p)}（缺文件）` : presetWeightLabel(p);
    opt.disabled = p.ok === false;
    sel.appendChild(opt);
  }
  if (!activeCampaign || !activeSimMode) {
    const custom = document.createElement("option");
    custom.value = "custom";
    custom.textContent = "自定义路径…";
    sel.appendChild(custom);
  }
  if (visibleIds.has(cur)) sel.value = cur;
  else if (visible.length) sel.value = visible[0].id;
  else sel.value = "custom";
}

function inferModelModeTag(preset, label) {
  if (preset?.readout_mode === "e1f" || /E1f|四成员/.test(String(label || ""))) {
    return "E1f 四成员在线融合";
  }
  if (preset?.id === "openbmi_baseline" || /OpenBMI/.test(String(label || ""))) {
    return "单模 Shallow（T0）";
  }
  if (preset?.kind === "ft_run" || preset?.kind === "current") {
    return "被试微调权重";
  }
  if (preset?.id === "custom" || /自定义/.test(String(label || ""))) {
    return "自定义路径";
  }
  return "";
}

function updateCurrentModelBanner() {
  if (!el.modelCurrentName) return;
  const presetId = el.modelPreset?.value || "";
  const task = el.s3TaskCkpt?.value?.trim() || "";
  const three = el.s3ThreeCkpt?.value?.trim() || "";
  let preset = null;
  if (presetId && presetId !== "custom") {
    preset = MODEL_PRESETS.find((x) => x.id === presetId) || null;
  }
  if (!preset) {
    preset = MODEL_PRESETS.find((p) => p.task === task && p.three === three) || null;
  }
  let modelName = preset?.model_name || ACTIVE_MODEL_LABEL || "";
  if (!modelName && preset) modelName = presetModelName(preset);
  if (!modelName) {
    modelName = shortWeightLabel(three || task);
    if (modelName && modelName !== "—") modelName = `自定义 · ${modelName}`;
  }
  if (!modelName || modelName === "—") modelName = "未配置";
  el.modelCurrentName.textContent = modelName;
  if (el.modelCurrentMode) {
    const modeTag = inferModelModeTag(preset, presetWeightLabel(preset));
    el.modelCurrentMode.textContent = modeTag ? `· ${modeTag}` : "";
  }
}

function fillModelPresets(presets, active) {
  MODEL_PRESETS = Array.isArray(presets) ? presets.slice() : [];
  const task = active?.task || el.s3TaskCkpt?.value || "";
  const three = active?.three || el.s3ThreeCkpt?.value || "";
  if (el.s3TaskCkpt && task) el.s3TaskCkpt.value = task;
  if (el.s3ThreeCkpt && three) el.s3ThreeCkpt.value = three;

  refreshModelPresetSelectOptions();

  const match =
    active?.preset_id ||
    MODEL_PRESETS.find((p) => p.task === task && p.three === three)?.id ||
    el.modelPreset?.value ||
    "custom";
  const visible = visiblePresetsForCampaign();
  const okId =
    visible.some((p) => p.id === match) || (!activeCampaign && match === "custom");
  if (el.modelPreset) el.modelPreset.value = okId ? match : (visible[0]?.id || "custom");

  updateWeightsStatusFromInputs();
  if (active?.model_label) ACTIVE_MODEL_LABEL = active.model_label;
  syncCampaignWeightLockUi();
}

function rejectCampaignPresetChange(revertId) {
  if (revertId && el.modelPreset) el.modelPreset.value = revertId;
  else applyCampaignLatestWeights();
  updateWeightsStatusFromInputs();
  syncCampaignWeightLockUi();
}

function findLatestPresetForCampaign(campaignId) {
  const cid = String(campaignId || "").trim();
  if (!cid) return null;
  const fts = MODEL_PRESETS.filter(
    (p) => p.kind === "ft_run" && p.campaign_id === cid && p.ok,
  );
  if (!fts.length) return null;
  const passed = fts.filter((p) => p.release_pass === true);
  return (passed.length ? passed : fts)[0];
}

function applyCampaignLatestWeights() {
  if (!activeCampaign?.campaign_id) return;
  const consumed = activeCampaign.runs_consumed || [];
  if (!consumed.length) {
    const r0 = campaignModelPicks[activeCampaign.campaign_id];
    const locked = inferCampaignLockedModelPresetId(activeCampaign);
    const pick = r0 || locked;
    if (pick && isZeroSamplePresetId(pick)) {
      if (el.modelPreset) el.modelPreset.value = pick;
      applyModelPreset(pick);
      return;
    }
    const e1f = MODEL_PRESETS.find((p) => p.id === "e1f_four_member" && p.ok !== false);
    const fallback = e1f ? "e1f_four_member" : "openbmi_baseline";
    if (el.modelPreset) el.modelPreset.value = fallback;
    applyModelPreset(fallback);
    return;
  }
  const hit = findLatestPresetForCampaign(activeCampaign.campaign_id);
  if (!hit) return;
  if (el.modelPreset) el.modelPreset.value = hit.id;
  applyModelPreset(hit.id);
}

function refreshModelPresetsFromServer(msg) {
  if (!msg?.model_presets?.length) return;
  let active = null;
  const outNorm = String(msg.out_dir || "").replace(/\\/g, "/");
  if (outNorm) {
    const hit = msg.model_presets.find((p) => {
      const t = String(p.task || "").replace(/\\/g, "/");
      return t && outNorm.includes(t.replace(/\/best_task\.pt$/, ""));
    });
    if (hit) active = { preset_id: hit.id, task: hit.task, three: hit.three };
  }
  if (!active && msg.weights?.task && msg.weights?.three) {
    active = {
      preset_id: msg.weights.preset_id,
      task: msg.weights.task,
      three: msg.weights.three,
    };
  }
  if (!active && msg.active_weights?.task && msg.active_weights?.three) {
    active = msg.active_weights;
  }
  fillModelPresets(msg.model_presets, active);
  if (!active && activeCampaign?.campaign_id) {
    applyCampaignLatestWeights();
  }
}

function applyModelPreset(id) {
  const p = MODEL_PRESETS.find((x) => x.id === id);
  if (!p) return;
  if (el.s3TaskCkpt) el.s3TaskCkpt.value = p.task;
  if (el.s3ThreeCkpt) el.s3ThreeCkpt.value = p.three;
  updateWeightsStatusFromInputs();
}

function shortWeightLabel(path) {
  const p = String(path || "").replace(/\\/g, "/");
  if (!p) return "—";
  const parts = p.split("/");
  const i = parts.indexOf("models");
  if (i >= 0 && parts[i + 1]) return parts[i + 1];
  if (/openbmi|5070_baseline/i.test(p)) return "openbmi_baseline";
  return parts.length >= 2 ? parts[parts.length - 2] : parts[parts.length - 1];
}

function updateWeightsStatusFromInputs() {
  const task = el.s3TaskCkpt?.value?.trim() || "";
  const three = el.s3ThreeCkpt?.value?.trim() || "";
  const presetId = el.modelPreset?.value || "";
  const preset = MODEL_PRESETS.find((p) => p.id === presetId) || null;
  const weightLabel = preset
    ? presetWeightLabel(preset)
    : shortWeightLabel(three || task);
  if (el.stWeights) {
    el.stWeights.textContent = three || task ? `${weightLabel}\nT: ${task}\n3: ${three}` : "—";
  }
  updateCurrentModelBanner();
  if (el.modelPreset && el.modelPreset.value !== "custom") {
    const hit = MODEL_PRESETS.find((p) => p.task === task && p.three === three);
    if (hit && presetAllowedForCampaign(hit, activeCampaign)) el.modelPreset.value = hit.id;
  }
}

function setWeightsDisplay(weights) {
  if (!weights) {
    updateWeightsStatusFromInputs();
    return;
  }
  const weightLabel = weights.weight_label || weights.label || shortWeightLabel(weights.three || weights.task);
  const task = weights.task || "";
  const three = weights.three || "";
  if (el.stWeights) {
    el.stWeights.textContent = `${weightLabel}\nT: ${task}\n3: ${three}`;
  }
  if (weights.model_label) ACTIVE_MODEL_LABEL = weights.model_label;
  updateCurrentModelBanner();
}

function readWeightOverrides() {
  const task = String(el.s3TaskCkpt?.value || "").trim();
  const three = String(el.s3ThreeCkpt?.value || "").trim();
  const out = {};
  if (task) out.s3_task_ckpt = task;
  if (three) out.s3_three_ckpt = three;
  // 与权重一并下发 readout，避免 UI 选 OpenBMI 但仍跑 yaml 默认 e1f
  const presetId = el.modelPreset?.value || "";
  let preset =
    presetId && presetId !== "custom"
      ? MODEL_PRESETS.find((p) => p.id === presetId) || null
      : null;
  if (!preset && task && three) {
    preset = MODEL_PRESETS.find((p) => p.task === task && p.three === three) || null;
  }
  if (preset?.readout_mode) {
    out.readout_mode = preset.readout_mode;
    if (preset.e1f_config_path) out.e1f_config_path = preset.e1f_config_path;
    if (preset.e1f_overlay_path) out.e1f_overlay_path = preset.e1f_overlay_path;
    if (preset.primary_judge_mode) {
      out.primary_judge_mode = preset.primary_judge_mode;
    } else if (String(preset.readout_mode).toLowerCase() === "e1f") {
      out.primary_judge_mode = "majority";
    } else {
      out.primary_judge_mode = "majority";
    }
  }
  return out;
}

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
  el.btnV3Guidance?.classList.remove("guidance-pulse");
  el.v2GuidanceCountdown?.classList.add("hidden");
  el.v2GuidanceCountdown?.classList.remove("urgent");
}

function startGuidanceCountdown(totalSec) {
  stopGuidanceCountdown();
  if (!totalSec || !el.v2GuidanceCountdown) return;
  let left = Math.ceil(Number(totalSec));
  const tick = () => {
    if (el.v2GuidanceCountdown) {
      el.v2GuidanceCountdown.textContent = `引导倒计时 ${left}s — 请完成双手分别抓握杯子引导后点「确认动觉引导完成」`;
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
  el.btnV3Guidance?.classList.add("guidance-pulse");
  playAlertBeep("guidance");
  tick();
  guidanceTimer = setInterval(tick, 1000);
}

const V23_TIMING_PRESETS = {
  openbmi: { prep_s: 2, cue_s: 0, imagine_s: 4, iti_s: 3, inter_trial_rest_s: 4 },
  legacy: { prep_s: 2, cue_s: 2, imagine_s: 6, iti_s: 3, inter_trial_rest_s: 0 },
};

/** @deprecated use V23_TIMING_PRESETS */
const V2_TIMING_PRESETS = V23_TIMING_PRESETS;

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
  ["v2_inter_trial_rest_s", "inter_trial_rest_s", "float"],
  ["v2_gate_enter_three", "gate_enter_three", "float"],
  ["v2_gate_min_quiz_trials", "gate_min_quiz_trials", "int"],
  ["v2_ft_min_valid_trials", "ft_min_valid_trials", "int"],
  ["v2_group_lr", "group_lr", "float"],
  ["v2_drift_patience", "drift_patience", "int"],
  ["v2_task_p_on", "task_p_on", "float"],
  ["v2_ft_epochs", "ft_epochs", "int"],
  ["v2_ft_batch_size", "ft_batch_size", "int"],
];

const DEFAULT_FT_REPLAY_RATIO = 0.10;

/** Exp29 启发式：~300 窗/run，约 4 run 在线 trial acc 可达 60% */
const REPLAY_ADVICE = {
  WINDOWS_PER_RUN: 300,
  RUNS_REPLAY_OFF: 4,
  RUNS_REPLAY_ON: 2,
  TOTAL_WINDOWS_OFF: 1200,
  TOTAL_WINDOWS_ON: 600,
  PRIMARY_ACC_OFF: 0.55,
};

let ftSessionCatalog = [];
/** 本场 session_saved 后一次性用于 FT 列表自动勾选（仅该路径） */
let lastFtAutoSelectRoot = null;
let replayAdviceManualOverride = { setup: false, panel: false };
let lastReplayAdvice = { setup: null, panel: null };

function normSessionPath(p) {
  return String(p || "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/\/+$/, "")
    .toLowerCase();
}

function sortFtSessionsForPanel(sessions, autoSelectRoot) {
  const autoNorm = autoSelectRoot ? normSessionPath(autoSelectRoot) : "";
  return (sessions || []).slice().sort((a, b) => {
    if (autoNorm) {
      const am = normSessionPath(a.path) === autoNorm;
      const bm = normSessionPath(b.path) === autoNorm;
      if (am !== bm) return am ? -1 : 1;
    }
    return String(b.dir || "").localeCompare(String(a.dir || ""));
  });
}

function sessionWindowEstimate(s) {
  if (!s) return REPLAY_ADVICE.WINDOWS_PER_RUN;
  if (s.n_windows != null && Number.isFinite(Number(s.n_windows))) {
    return Number(s.n_windows);
  }
  if (s.n_windows_est != null && Number.isFinite(Number(s.n_windows_est))) {
    return Number(s.n_windows_est);
  }
  if (s.n_trials != null && Number.isFinite(Number(s.n_trials))) {
    return Number(s.n_trials) * 11;
  }
  return REPLAY_ADVICE.WINDOWS_PER_RUN;
}

function estimatePlannedSessionWindows() {
  if (isV3Mode()) {
    const blocks = Number(el.form.elements.namedItem("v3_blocks")?.value) || 2;
    const tpb = Number(el.form.elements.namedItem("v3_trials_per_block")?.value) || 18;
    return blocks * tpb * 11;
  }
  if (isV2Mode()) {
    const rounds = Number(el.form.elements.namedItem("v2_cal_rounds_max")?.value) || 6;
    const ftTrials = Number(el.form.elements.namedItem("v2_ft_trials_per_round")?.value) || 12;
    const gameR = Number(el.form.elements.namedItem("v2_game_rounds")?.value) || 2;
    const gameT = Number(el.form.elements.namedItem("v2_game_trials_per_round")?.value) || 16;
    return (rounds * ftTrials + gameR * gameT) * 11;
  }
  return REPLAY_ADVICE.WINDOWS_PER_RUN;
}

function computeReplayAdvice(selectedSessions, { includePlanned = false } = {}) {
  const sessions = Array.isArray(selectedSessions) ? selectedSessions : [];
  let totalWindows = sessions.reduce((sum, s) => sum + sessionWindowEstimate(s), 0);
  const nSel = sessions.length;
  if (includePlanned) {
    totalWindows += estimatePlannedSessionWindows();
  }
  const nTotal = nSel + (includePlanned ? 1 : 0);

  const accs = sessions
    .map((s) => s.primary_acc)
    .filter((v) => v != null && Number.isFinite(Number(v)))
    .map(Number);
  const avgAcc = accs.length ? accs.reduce((a, b) => a + b, 0) / accs.length : null;

  let recommend = true;
  let level = "optional";
  let reason = "";

  if (
    nTotal >= REPLAY_ADVICE.RUNS_REPLAY_OFF ||
    totalWindows >= REPLAY_ADVICE.TOTAL_WINDOWS_OFF
  ) {
    recommend = false;
    level = "off";
    reason =
      `已选 ${nTotal} 个 session、约 ${Math.round(totalWindows)} 训练窗` +
      "（Exp29：~300 窗/run × 4 run 可达 60%）→ 建议关闭 replay，纯被试窗 FT";
  } else if (
    nTotal <= REPLAY_ADVICE.RUNS_REPLAY_ON &&
    totalWindows < REPLAY_ADVICE.TOTAL_WINDOWS_ON
  ) {
    recommend = true;
    level = "on";
    reason =
      `约 ${Math.round(totalWindows)} 窗 / ${nTotal} session` +
      " → 数据偏少，建议开启 T0 replay（0.10）";
  } else {
    recommend = true;
    level = "optional";
    reason =
      `约 ${Math.round(totalWindows)} 窗 / ${nTotal} session` +
      " → 默认可开 replay；若在线 acc 已稳定 ≥55% 可试关";
  }

  if (
    avgAcc != null &&
    avgAcc >= REPLAY_ADVICE.PRIMARY_ACC_OFF &&
    nTotal >= 3
  ) {
    recommend = false;
    level = "off";
    reason +=
      `；valid-primary 均值 ${(avgAcc * 100).toFixed(0)}% 偏高，更倾向关 replay`;
  }

  return {
    recommend_use_replay: recommend,
    level,
    reason,
    total_windows: totalWindows,
    n_sessions: nTotal,
    avg_primary_acc: avgAcc,
  };
}

function renderReplayAdviceHint(hintEl, advice) {
  if (!hintEl || !advice) return;
  hintEl.textContent = advice.reason;
  hintEl.classList.remove("replay-advice-on", "replay-advice-off", "replay-advice-optional");
  hintEl.classList.add(
    advice.level === "off"
      ? "replay-advice-off"
      : advice.level === "on"
        ? "replay-advice-on"
        : "replay-advice-optional",
  );
}

function applyReplayAdvice(advice, scope, { force = false } = {}) {
  if (!advice) return;
  const key = scope === "panel" ? "panel" : "setup";
  if (!force && replayAdviceManualOverride[key]) return;
  const useReplay = advice.recommend_use_replay;
  if (scope === "panel") {
    if (el.ftUseReplay) el.ftUseReplay.checked = useReplay;
  } else {
    const setupUse = el.form?.querySelector('[name="ft_use_replay"]');
    if (setupUse) setupUse.checked = useReplay;
    if (el.ftUseReplay) el.ftUseReplay.checked = useReplay;
  }
  syncFtReplayRatioUi();
}

function updateReplayAdvice(scope, { autoApply = null } = {}) {
  const isPanel = scope === "panel";
  if (autoApply === null) autoApply = isPanel;
  let advice;
  if (isPanel) {
    const selected = collectSelectedFtSessions();
    advice = computeReplayAdvice(selected, { includePlanned: false });
    lastReplayAdvice.panel = advice;
    renderReplayAdviceHint(el.ftReplayHint, advice);
    if (el.btnFtReplayAdopt) el.btnFtReplayAdopt.classList.remove("hidden");
    if (autoApply) applyReplayAdvice(advice, "panel");
  } else {
    const past = (activeSubjectInfo?.sessions || []).filter((s) => s.ft_eligible);
    advice = computeReplayAdvice(past, { includePlanned: true });
    lastReplayAdvice.setup = advice;
    renderReplayAdviceHint(el.setupFtReplayHint, advice);
    if (el.btnSetupReplayAdopt) el.btnSetupReplayAdopt.classList.remove("hidden");
    if (autoApply) applyReplayAdvice(advice, "setup");
  }
}

function readFtAdvancedOptions() {
  const setupEarly = el.form?.querySelector('[name="ft_early_stop"]');
  const setupMax = el.form?.elements?.namedItem("ft_max_epochs");
  const setupPat = el.form?.elements?.namedItem("ft_patience");
  const setupFixed = el.form?.elements?.namedItem("ft_fixed_epochs");
  const setupDet = el.form?.querySelector('[name="ft_deterministic"]');
  const setupSeed = el.form?.elements?.namedItem("ft_seed");

  const earlyStop = el.ftEarlyStop ? el.ftEarlyStop.checked : Boolean(setupEarly?.checked ?? true);
  const maxEpochs = Number(
    el.ftMaxEpochs?.value ?? setupMax?.value ?? 20,
  );
  const patience = Number(el.ftPatience?.value ?? setupPat?.value ?? 5);
  const fixedEpochs = Number(el.ftFixedEpochs?.value ?? setupFixed?.value ?? 5);
  const deterministic = el.ftDeterministic
    ? el.ftDeterministic.checked
    : Boolean(setupDet?.checked ?? true);
  const seed = Number(el.ftSeed?.value ?? setupSeed?.value ?? 42);

  return {
    early_stop: earlyStop,
    max_epochs: Number.isFinite(maxEpochs) ? maxEpochs : 20,
    patience: Number.isFinite(patience) ? patience : 5,
    fixed_epochs: Number.isFinite(fixedEpochs) ? fixedEpochs : 5,
    deterministic,
    seed: Number.isFinite(seed) ? seed : 42,
  };
}

function syncFtAdvancedUi() {
  const adv = readFtAdvancedOptions();
  const disFixed = adv.early_stop;
  if (el.ftFixedEpochs) el.ftFixedEpochs.disabled = disFixed;
  const setupFixed = el.form?.elements?.namedItem("ft_fixed_epochs");
  if (setupFixed) setupFixed.disabled = disFixed;
  if (el.ftMaxEpochs) el.ftMaxEpochs.disabled = !adv.early_stop;
  if (el.ftPatience) el.ftPatience.disabled = !adv.early_stop;
  const setupMax = el.form?.elements?.namedItem("ft_max_epochs");
  const setupPat = el.form?.elements?.namedItem("ft_patience");
  if (setupMax) setupMax.disabled = !adv.early_stop;
  if (setupPat) setupPat.disabled = !adv.early_stop;
}

function applyFtAdvancedDefaults(ftDefaults) {
  const d = ftDefaults || {};
  const earlyStop = d.early_stop !== false;
  const maxEp = d.max_epochs != null ? Number(d.max_epochs) : 20;
  const pat = d.patience != null ? Number(d.patience) : 5;
  const fixed = d.fixed_epochs != null ? Number(d.fixed_epochs) : 5;
  const det = d.deterministic !== false;
  const seed = d.seed != null ? Number(d.seed) : 42;

  if (el.ftEarlyStop) el.ftEarlyStop.checked = earlyStop;
  if (el.ftMaxEpochs) el.ftMaxEpochs.value = String(maxEp);
  if (el.ftPatience) el.ftPatience.value = String(pat);
  if (el.ftFixedEpochs) el.ftFixedEpochs.value = String(fixed);
  if (el.ftDeterministic) el.ftDeterministic.checked = det;
  if (el.ftSeed) el.ftSeed.value = String(seed);

  const setupEarly = el.form?.querySelector('[name="ft_early_stop"]');
  const setupMax = el.form?.elements?.namedItem("ft_max_epochs");
  const setupPat = el.form?.elements?.namedItem("ft_patience");
  const setupFixed = el.form?.elements?.namedItem("ft_fixed_epochs");
  const setupDet = el.form?.querySelector('[name="ft_deterministic"]');
  const setupSeed = el.form?.elements?.namedItem("ft_seed");
  if (setupEarly) setupEarly.checked = earlyStop;
  if (setupMax) setupMax.value = String(maxEp);
  if (setupPat) setupPat.value = String(pat);
  if (setupFixed) setupFixed.value = String(fixed);
  if (setupDet) setupDet.checked = det;
  if (setupSeed) setupSeed.value = String(seed);
  syncFtAdvancedUi();
}

function readFtReplayOptions(scope) {
  if (scope === el.form) {
    const useNode = scope.querySelector('[name="ft_use_replay"]');
    const ratioNode = scope.elements?.namedItem("ft_replay_ratio");
    const useReplay = useNode ? useNode.checked : true;
    const ratioRaw = ratioNode ? Number(ratioNode.value) : DEFAULT_FT_REPLAY_RATIO;
    const ratio = Number.isFinite(ratioRaw) ? ratioRaw : DEFAULT_FT_REPLAY_RATIO;
    return {
      use_replay: useReplay,
      replay_ratio: useReplay ? ratio : 0,
    };
  }
  const useReplay = el.ftUseReplay ? el.ftUseReplay.checked : true;
  const ratioRaw = el.ftReplayRatio ? Number(el.ftReplayRatio.value) : DEFAULT_FT_REPLAY_RATIO;
  const ratio = Number.isFinite(ratioRaw) ? ratioRaw : DEFAULT_FT_REPLAY_RATIO;
  return {
    use_replay: useReplay,
    replay_ratio: useReplay ? ratio : 0,
  };
}

function syncFtReplayRatioUi() {
  const setup = readFtReplayOptions(el.form);
  const setupRatio = el.form?.elements?.namedItem("ft_replay_ratio");
  if (setupRatio) setupRatio.disabled = !setup.use_replay;
  if (el.ftReplayRatio) el.ftReplayRatio.disabled = !readFtReplayOptions(null).use_replay;
}

function applyFtReplayDefaults(ftDefaults) {
  const d = ftDefaults || {};
  const useReplay = d.use_replay !== false;
  const ratio =
    d.replay_ratio != null && Number.isFinite(Number(d.replay_ratio))
      ? Number(d.replay_ratio)
      : DEFAULT_FT_REPLAY_RATIO;
  const setupUse = el.form?.querySelector('[name="ft_use_replay"]');
  const setupRatio = el.form?.elements?.namedItem("ft_replay_ratio");
  if (setupUse) setupUse.checked = useReplay;
  if (setupRatio) setupRatio.value = String(ratio);
  if (el.ftUseReplay) el.ftUseReplay.checked = useReplay;
  if (el.ftReplayRatio) el.ftReplayRatio.value = String(ratio);
  syncFtReplayRatioUi();
}

function readV23TimingFromForm(prefix) {
  const prep = Number(el.form.elements.namedItem(`${prefix}_prep_s`)?.value) || 0;
  const cue = Number(el.form.elements.namedItem(`${prefix}_cue_s`)?.value) || 0;
  const mi = Number(el.form.elements.namedItem(`${prefix}_imagine_s`)?.value) || 0;
  const iti = Number(el.form.elements.namedItem(`${prefix}_iti_s`)?.value) || 0;
  const rest = Number(el.form.elements.namedItem(`${prefix}_inter_trial_rest_s`)?.value) || 0;
  return {
    fixation_s: prep,
    cue_s: cue,
    mi_s: mi,
    post_mi_hold_s: 0,
    rest_s: rest,
    transition_s: iti,
  };
}

function readV2TimingFromForm() {
  return readV23TimingFromForm("v2");
}

function readV3TimingFromForm() {
  return readV23TimingFromForm("v3");
}

function applyV23TimingPreset(prefix, presetName) {
  const p = V23_TIMING_PRESETS[presetName];
  if (!p) return;
  for (const [k, v] of Object.entries(p)) {
    const node = el.form.elements.namedItem(`${prefix}_${k}`);
    if (node) node.value = v;
  }
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
  Object.assign(o, readWeightOverrides());
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
  const ftRep = readFtReplayOptions(el.form);
  o.replay_ratio = ftRep.replay_ratio;
  Object.assign(o, readWeightOverrides());
  return o;
}

function applyV3OverridesToForm(ov) {
  if (!ov) return;
  for (const [formName, key] of V3_OVERRIDE_KEYS) {
    if (ov[key] == null) continue;
    const node = el.form.elements.namedItem(formName);
    if (node) node.value = ov[key];
  }
  renderSetupTimelineV3();
}

function applyV2OverridesToForm(ov) {
  if (!ov) return;
  for (const [formName, key] of V2_OVERRIDE_KEYS) {
    if (ov[key] == null) continue;
    const node = el.form.elements.namedItem(formName);
    if (node) node.value = ov[key];
  }
  if (ov.replay_ratio != null) {
    const useReplay = Number(ov.replay_ratio) > 0;
    const setupUse = el.form?.querySelector('[name="ft_use_replay"]');
    const setupRatio = el.form?.elements?.namedItem("ft_replay_ratio");
    if (setupUse) setupUse.checked = useReplay;
    if (setupRatio) setupRatio.value = String(ov.replay_ratio);
  }
  syncFtReplayRatioUi();
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
  const { total } = renderTimeline(el.setupTimelineV2, t, V23_TIMELINE_SEGMENTS) || { total: 0 };
  updateV2TrialComposeHint();
  if (el.timingHintV2) {
    const calMin = Number(el.form.elements.namedItem("v2_cal_rounds_min")?.value) || 4;
    const calMax = Number(el.form.elements.namedItem("v2_cal_rounds_max")?.value) || 6;
    const gameR = Number(el.form.elements.namedItem("v2_game_rounds")?.value) || 2;
    const perCal = Number(el.form.elements.namedItem("v2_trials_per_round")?.value) || 18;
    const perGame = Number(el.form.elements.namedItem("v2_game_trials_per_round")?.value) || 16;
    const trials = calMax * perCal + gameR * perGame;
    const est = trials * total;
    const cue = Number(el.form.elements.namedItem("v2_cue_s")?.value) || 0;
    el.timingHintV2.textContent =
      `单 trial = ${total}s（OpenBMI-Align${cue <= 0 ? " · Cue=MI onset" : ""}）· 标定 ${calMin}–${calMax} 轮×${perCal} + 游戏 ${gameR}×${perGame}` +
      ` · 纯试次 ≈ ${Math.round(est / 60)} 分钟（含 ITI/试次间 Rest，不含块间休息）`;
  }
}

function renderSetupTimelineV3() {
  const t = readV3TimingFromForm();
  const { total } = renderTimeline(el.setupTimelineV3, t, V23_TIMELINE_SEGMENTS) || { total: 0 };
  if (el.timingHintV3) {
    const blocks = Number(el.form.elements.namedItem("v3_blocks")?.value) || 2;
    const tpb = Number(el.form.elements.namedItem("v3_trials_per_block")?.value) || 18;
    const baseline = v3BaselineRestS();
    const gap = Number(el.form.elements.namedItem("v3_block_gap_s")?.value) || 90;
    const trials = blocks * tpb;
    const cue = Number(el.form.elements.namedItem("v3_cue_s")?.value) || 0;
    el.timingHintV3.textContent =
      `单 trial = ${total}s（OpenBMI-Align${cue <= 0 ? " · Cue=MI onset" : ""}）· ${blocks} 块×${tpb} 试次` +
      ` · 块间 ${gap}s + 开场基线 ${baseline}s · 纯试次 ≈ ${Math.round((trials * total) / 60)} 分钟`;
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

function isSimV3Mode() {
  return (
    Boolean(activeSimMode) &&
    (el.form.querySelector('input[name="phase_mode"]:checked') || {}).value === "sim_v3_session"
  );
}

function isV3Mode() {
  const pm = (el.form.querySelector('input[name="phase_mode"]:checked') || {}).value;
  if (pm === "v3_session") return true;
  return isSimV3Mode();
}

/** 真机工作区不应保留 sim_v3（否则会隐藏采集板卡并误显 BCI2a 块） */
function ensureExperimentPhaseMode() {
  if (activeSimMode) return;
  const checked = el.form?.querySelector('input[name="phase_mode"]:checked');
  if (checked?.value !== "sim_v3_session") return;
  const v3 = el.form?.querySelector('input[name="phase_mode"][value="v3_session"]');
  if (v3) {
    v3.checked = true;
    return;
  }
  const v2 = el.form?.querySelector('input[name="phase_mode"][value="v2_session"]');
  if (v2) v2.checked = true;
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
      v:
        m.active_channels != null
          ? `${m.active_channels}/${m.n_scoring_channels != null ? m.n_scoring_channels : 8}`
          : "—",
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
  document.getElementById("btn-gate")?.classList.toggle("hidden", on);
  document.getElementById("btn-split")?.classList.toggle("hidden", on || isV2Mode());
}

function syncWorkModeUi() {
  const simWorkspace = Boolean(activeSimMode);
  document.querySelectorAll(".exp-only").forEach((node) => {
    node.classList.toggle("hidden", simWorkspace);
  });
  document.querySelectorAll(".exp-phase-only").forEach((node) => {
    node.classList.toggle("hidden", simWorkspace);
  });
  document.querySelectorAll(".sim-phase-only").forEach((node) => {
    node.classList.toggle("hidden", !simWorkspace);
  });
  document.querySelectorAll(".sim-workspace-only").forEach((node) => {
    node.classList.toggle("hidden", !simWorkspace);
  });
  if (simWorkspace) {
    const simRadio = el.form?.querySelector('input[name="phase_mode"][value="sim_v3_session"]');
    if (simRadio) simRadio.checked = true;
    const acq = el.form?.querySelector('[name="acq_enabled"]');
    if (acq) acq.checked = true;
    const openPage = el.form?.querySelector('[name="open_subject_page"]');
    if (openPage) {
      openPage.checked = true;
      openPage.disabled = true;
    }
    const fb = el.form?.querySelector('[name="subject_feedback_mode"]');
    if (fb) {
      fb.value = "arm_reach";
      fb.disabled = true;
    }
    if (el.setupSessionId) {
      el.setupSessionId.disabled = true;
      el.setupSessionId.title = "仿真模式：会话编号与 run 一致，跑完自动切换";
    }
    if (el.sessionIdSuggestBtn) el.sessionIdSuggestBtn.disabled = true;
    refreshSimRunSuggestion();
  } else {
    ensureExperimentPhaseMode();
    const openPage = el.form?.querySelector('[name="open_subject_page"]');
    if (openPage) openPage.disabled = false;
    const fb = el.form?.querySelector('[name="subject_feedback_mode"]');
    if (fb) fb.disabled = false;
    if (el.setupSessionId) {
      el.setupSessionId.disabled = false;
      el.setupSessionId.title = "";
    }
    if (el.sessionIdSuggestBtn) el.sessionIdSuggestBtn.disabled = false;
  }
  syncPhaseModeUi();
}

function syncPhaseModeUi() {
  const v2 = isV2Mode();
  const v3 = isV3Mode();
  const sim = isSimV3Mode();
  const v4 = isV4Mode();
  document.querySelectorAll(".phase2-only").forEach((node) => {
    node.classList.toggle("hidden", v2 || v3 || v4);
  });
  document.querySelectorAll(".v2-only").forEach((node) => {
    node.classList.toggle("hidden", !v2);
  });
  document.querySelectorAll(".v3-only").forEach((node) => {
    node.classList.toggle("hidden", !v3 || sim);
  });
  document.querySelectorAll(".sim-only").forEach((node) => {
    node.classList.toggle("hidden", !sim);
  });
  document.querySelectorAll(".v23-only").forEach((node) => {
    node.classList.toggle("hidden", !(v2 || v3));
  });
  document.querySelectorAll(".v4-only").forEach((node) => {
    node.classList.toggle("hidden", !v4);
  });
  syncProtocolLockUi();
  if (v2) renderSetupTimelineV2();
  else if (v3) renderSetupTimelineV3();
  else if (!v4) renderSetupTimeline();
  if (v2 || v3) updateReplayAdvice("setup", { autoApply: false });
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
      el.v3PowerBars.innerHTML = v3BaselineWaitHtml();
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
  document.getElementById("st-session-score-wrap")?.classList.toggle("hidden", !on);
  document.getElementById("p4-summary-v1")?.classList.toggle("hidden", on || isV2Mode());
  el.p4SummaryV2?.classList.toggle("hidden", true);
  document.getElementById("btn-gate")?.classList.toggle("hidden", on);
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
  document.getElementById("st-session-score-wrap")?.classList.toggle("hidden", !on);
  document.getElementById("p4-summary-v1")?.classList.toggle("hidden", on);
  el.p4SummaryV2?.classList.toggle("hidden", !on);
  document.getElementById("btn-gate")?.classList.toggle("hidden", !on);
  document.getElementById("btn-v2-enter-game")?.classList.toggle("hidden", !on);
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

function updateV2Progress(prog, score, msg) {
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
  // 本试次：多数票 0/1（仅 trial_end 更新，避免 judge 窗内 arm 分干扰）
  let trialSc = null;
  if (msg?.stage === "trial_end") {
    const sum = msg.data?.summary;
    if (sum?.score != null) trialSc = sum.score;
    else if (score != null) trialSc = score;
    else if (prog?.score != null) trialSc = prog.score;
  }
  if (trialSc != null) {
    const n = Number(trialSc);
    const txt = Number.isInteger(n) ? String(n) : String(Math.round(n));
    if (el.stV2Score) el.stV2Score.textContent = txt;
    if (el.v2ScoreNum) el.v2ScoreNum.textContent = txt;
    if (el.v2ScoreFill) el.v2ScoreFill.style.width = n >= 1 ? "100%" : "0%";
  }
  applySessionScore(msg || { progress: prog });
  if (prog.phase_step) setPhaseStepV2(prog.phase_step);
}

function updateV2Gate(msg) {
  if (el.v2GateAcc) {
    el.v2GateAcc.textContent =
      msg.acc != null ? `${(Number(msg.acc) * 100).toFixed(1)}%` : "—";
  }
  if (el.v2GateStatus) {
    const adv = msg.advisory || msg.awaiting_operator ? "（建议）" : "";
    el.v2GateStatus.textContent = `${msg.status || "—"}${adv}`;
  }
  if (el.v2GateN) el.v2GateN.textContent = String(msg.n_quiz ?? 0);
  if (el.v2GateCurve && Array.isArray(msg.curve)) {
    el.v2GateCurve.textContent = msg.curve.map(([k, a]) => `k=${k}: ${(a * 100).toFixed(1)}%`).join("\n");
  }
  el.v2WeakMi?.classList.toggle("hidden", msg.status !== "weak_mi");
  if (msg.progress) updateV2Progress(msg.progress, msg.score, msg);
  else applySessionScore(msg);
  setPhaseStepV2("gate");
  if (msg.awaiting_operator) {
    showRunAlert("等待操作员确认准入（G / 准入按钮）", "gate-pass");
  }
}

el.form.querySelectorAll('input[name="phase_mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    syncPhaseModeUi();
    if (!activeSimMode) {
      const sug = syncSuggestSessionIdForBoard();
      fillSessionIdSelect(sug);
      if (sug) setSessionIdSelectValue(sug, { overwrite: false });
    }
  });
});
let priorModelPresetId = "";

el.modelPreset?.addEventListener("focus", () => {
  priorModelPresetId = el.modelPreset?.value || "";
});
el.modelPreset?.addEventListener("change", () => {
  const id = el.modelPreset.value;
  const prevId = priorModelPresetId;
  priorModelPresetId = id;
  if (activeCampaign && activeSimMode) {
    if (id === "custom") {
      alert("Campaign 模式下不可使用自定义路径");
      rejectCampaignPresetChange(prevId);
      return;
    }
    const p = MODEL_PRESETS.find((x) => x.id === id);
    if (p && !presetAllowedForCampaign(p, activeCampaign)) {
      alert("不可使用其他 Campaign 的权重，或与本周期锁定的模型不一致");
      rejectCampaignPresetChange(prevId);
      return;
    }
    const consumed = activeCampaign.runs_consumed || [];
    const locked = inferCampaignLockedModelPresetId(activeCampaign);
    if (consumed.length && locked && isZeroSamplePresetId(id) && id !== locked) {
      alert(`本 Campaign 已锁定模型，不可切换为零样本 ${id}`);
      rejectCampaignPresetChange(prevId);
      return;
    }
    if (!consumed.length && isZeroSamplePresetId(id)) {
      campaignModelPicks[activeCampaign.campaign_id] = id;
    }
  }
  if (id && id !== "custom") applyModelPreset(id);
  else updateWeightsStatusFromInputs();
  syncCampaignWeightLockUi();
});
el.s3TaskCkpt?.addEventListener("input", () => {
  if (activeCampaign && activeSimMode) return;
  if (el.modelPreset) el.modelPreset.value = "custom";
  updateWeightsStatusFromInputs();
});
el.s3ThreeCkpt?.addEventListener("input", () => {
  if (activeCampaign && activeSimMode) return;
  if (el.modelPreset) el.modelPreset.value = "custom";
  updateWeightsStatusFromInputs();
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
    applyV23TimingPreset("v2", btn.dataset.v2TimingPreset);
    renderSetupTimelineV2();
  });
});

document.querySelectorAll("[data-v3-timing-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    applyV23TimingPreset("v3", btn.dataset.v3TimingPreset);
    renderSetupTimelineV3();
  });
});

// 与 trial_v2 / session 执行序一致：Cue前 Rest → prep → cue → MI → ITI
const V23_TIMELINE_SEGMENTS = [
  { key: "rest_s", zh: "Cue前 Rest", cls: "tl-rest" },
  { key: "fixation_s", zh: "prep", cls: "tl-fixation" },
  { key: "cue_s", zh: "Cue", cls: "tl-cue" },
  { key: "mi_s", zh: "MI", cls: "tl-mi" },
  { key: "transition_s", zh: "ITI", cls: "tl-transition" },
];

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

/** 渲染时间轴：按各阶段秒数比例着色分段；segments 默认 phase2 TIMING_KEYS。
 *  cue_s≤0 时仍在 prep 后画「Cue·onset」标记（不计入合计秒数）。
 */
function renderTimeline(container, timing, segments) {
  if (!container) return null;
  let order;
  if (segments && segments.length && segments[0].key) {
    order = segments.map(({ key, zh, cls }) => ({
      key,
      zh,
      cls,
      s: Number(timing[key]) || 0,
    }));
  } else {
    const keys = segments || TIMING_KEYS;
    order = keys.map(([, key, zh, cls]) => ({
      key,
      zh,
      cls,
      s: Number(timing[key]) || 0,
    }));
  }
  const total = order.reduce((a, b) => a + b.s, 0);
  container.innerHTML = "";
  if (total <= 0 && !order.some((seg) => seg.key === "cue_s")) return { total };

  for (const seg of order) {
    // Cue：时长>0 正常块；时长=0 仍标在 prep 后（OpenBMI Cue=MI onset）
    if (seg.key === "cue_s" && seg.s <= 0) {
      const mark = document.createElement("div");
      mark.className = "tl-seg tl-cue tl-cue-onset";
      mark.title = "Cue = MI onset（prep 结束后立刻给提示，无单独 cue 时长）";
      const label = document.createElement("span");
      label.textContent = "Cue";
      mark.appendChild(label);
      const sub = document.createElement("small");
      sub.textContent = "onset";
      mark.appendChild(sub);
      container.appendChild(mark);
      continue;
    }
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
  const cueS = Number(timing.cue_s) || 0;
  cap.textContent =
    cueS <= 0
      ? `单 trial 合计 ${total}s（Cue 在 prep 后 · 与 MI 同刻 onset）`
      : `单 trial 合计 ${total}s`;
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
let activeSubject = null;
let activeSubjectInfo = null;
let activeSimMode = false;
let simCatalogRuns = [];
let simCampaigns = [];
let activeCampaign = null;
let lastFtRunDir = null;
let lastFtGatePass = null;
let ftBusy = false;
let sessionRecordExcluded = false;
let lastCampaignManifest = null;

function loadSubjectLoginLocal() {
  try {
    return JSON.parse(localStorage.getItem(SUBJECT_LOGIN_KEY) || "null");
  } catch {
    return null;
  }
}

function saveSubjectLoginLocal(info) {
  if (!info?.subject_id) return;
  localStorage.setItem(
    SUBJECT_LOGIN_KEY,
    JSON.stringify({
      subject_id: info.subject_id,
      display_name: info.subject?.display_name || "",
      sim_mode: Boolean(info.sim_mode),
    }),
  );
}

function campaignNextRun(manifest) {
  if (!manifest) return null;
  const consumed = new Set(manifest.runs_consumed || []);
  for (const r of manifest.session_queue || []) {
    if (!consumed.has(r)) return r;
  }
  return null;
}

function resolveSimRunId(useCampaignQueue = false) {
  if (activeCampaign) {
    const nxt = campaignNextRun(activeCampaign);
    if (nxt) return nxt;
    if (useCampaignQueue) return "";
  }
  const sug = String(activeSubjectInfo?.suggest_session_id || "").trim().toLowerCase();
  if (sug.startsWith("run")) return sug;
  const runInp = el.form?.querySelector('[name="sim_run_id"]');
  const fromInput = String(runInp?.value || "").trim().toLowerCase();
  if (fromInput.startsWith("run")) return fromInput;
  return "run3";
}

function applySimRunToForm(runId) {
  const rid = String(runId || "").trim().toLowerCase();
  if (!rid.startsWith("run")) return;
  const runInp = el.form?.querySelector('[name="sim_run_id"]');
  if (runInp) runInp.value = rid;
  if (el.setupSessionId) el.setupSessionId.value = rid;
}

function syncSimRunFromCampaignOrSuggest() {
  if (!activeSimMode && !isSimV3Mode()) return;
  const useQueue = Boolean(el.form?.querySelector('[name="sim_use_campaign_queue"]')?.checked);
  const runInp = el.form?.querySelector('[name="sim_run_id"]');
  const nxt = activeCampaign ? campaignNextRun(activeCampaign) : null;
  const rid = nxt || activeSubjectInfo?.suggest_session_id || null;
  if (rid) applySimRunToForm(rid);
  if (runInp) runInp.disabled = Boolean(useQueue && activeCampaign);
}

function refreshSimRunSuggestion() {
  if (!activeSimMode && !isSimV3Mode()) return;
  syncSimRunFromCampaignOrSuggest();
  if (activeCampaign) updateSimCampaignStatus();
  else if (el.simCampaignStatus) el.simCampaignStatus.textContent = "";
}

function requestSimCatalog() {
  if (!activeSimMode || !activeSubject) return;
  send({ type: "sim_catalog", subject_id: activeSubject });
}

function requestSimCampaignList() {
  if (!activeSimMode || !activeSubject) return;
  send({ type: "sim_campaign_list", subject_id: activeSubject });
}

function renderSimRunQueue(runs, consumed = []) {
  if (!el.simRunQueue) return;
  const used = new Set(consumed || []);
  const list = runs || [];
  if (!list.length) {
    el.simRunQueue.textContent = "暂无 run（检查 DATA/bci2a/AxxT.mat）";
    return;
  }
  el.simRunQueue.innerHTML = list
    .map((r) => {
      const rid = r.run_id || r;
      const done = used.has(rid);
      const shard = r.shard_ok ? "shard✓" : "shard—";
      const total = r.n_total_trials ?? r.n_lr_trials;
      const detail = r.n_rest_trials != null
        ? `Rest${r.n_rest_trials} L${r.n_left ?? "—"} R${r.n_right ?? "—"} · 共${total}`
        : `${r.n_lr_trials ?? "—"} L/R`;
      return `<label class="check sim-run-row${done ? " muted" : ""}">
        <input type="checkbox" class="sim-run-cb" value="${rid}" ${done ? "disabled" : ""} />
        ${rid} · ${detail} · ${shard}${done ? " · 已用" : ""}
      </label>`;
    })
    .join("");
}

function sortCampaignsNewestFirst(campaigns) {
  return [...(campaigns || [])].sort((a, b) =>
    String(b.campaign_id || "").localeCompare(String(a.campaign_id || "")),
  );
}

function fillSimCampaignSelect(campaigns, selectedId = "") {
  if (!el.simCampaignSelect) return;
  const prev = selectedId || el.simCampaignSelect.value || activeCampaign?.campaign_id || "";
  el.simCampaignSelect.innerHTML = '<option value="">— 单场 run —</option>';
  for (const c of sortCampaignsNewestFirst(campaigns)) {
    const rem = campaignRemainingRuns(c);
    if (rem.length === 0) continue;
    const opt = document.createElement("option");
    opt.value = c.campaign_id || "";
    opt.dataset.manifestPath = c.manifest_path || "";
    opt.textContent = `${c.campaign_id} · 剩余 ${rem.length}/${(c.session_queue || []).length}`;
    el.simCampaignSelect.appendChild(opt);
  }
  const stillThere = [...el.simCampaignSelect.options].some((o) => o.value === prev);
  if (prev && stillThere) {
    el.simCampaignSelect.value = prev;
    onSimCampaignSelectChange(false);
  } else if (prev && !stillThere) {
    el.simCampaignSelect.value = "";
    activeCampaign = null;
    syncCampaignWeightLockUi();
  }
}

function updateSimCampaignStatus() {
  if (!activeCampaign) {
    if (el.simCampaignStatus) el.simCampaignStatus.textContent = "";
    syncSimRunFromCampaignOrSuggest();
    return;
  }
  const consumed = activeCampaign.runs_consumed || [];
  const nxt = campaignNextRun(activeCampaign);
  if (el.simCampaignStatus) {
    el.simCampaignStatus.textContent =
      `Campaign ${activeCampaign.campaign_id} · 已用 ${consumed.join(", ") || "—"} · 下一场 ${nxt || "（已完成）"}`;
  }
  renderSimRunQueue(simCatalogRuns, consumed);
  syncSimRunFromCampaignOrSuggest();
}

function onSimCampaignSelectChange(applyWeights = true) {
  const sel = el.simCampaignSelect;
  if (!sel || sel.value === "") {
    activeCampaign = null;
    updateSimCampaignStatus();
    syncCampaignWeightLockUi();
    return;
  }
  activeCampaign = (simCampaigns || []).find((c) => c.campaign_id === sel.value) || null;
  updateSimCampaignStatus();
  if (applyWeights) applyCampaignLatestWeights();
  syncCampaignWeightLockUi();
}

function onSimCatalogAck(msg) {
  if (!msg.ok) {
    if (el.simRunQueue) el.simRunQueue.textContent = msg.message || "加载 run 失败";
    return;
  }
  simCatalogRuns = msg.runs || [];
  if (activeCampaign) updateSimCampaignStatus();
  else renderSimRunQueue(simCatalogRuns, []);
}

function onSimCampaignAck(msg) {
  if (!msg.ok) {
    alert(msg.message || "创建 Campaign 失败");
    return;
  }
  activeCampaign = msg.manifest || null;
  if (activeCampaign) {
    simCampaigns = [activeCampaign, ...simCampaigns.filter((c) => c.campaign_id !== activeCampaign.campaign_id)];
    fillSimCampaignSelect(simCampaigns, activeCampaign.campaign_id);
    const useQ = el.form?.querySelector('[name="sim_use_campaign_queue"]');
    if (useQ) useQ.checked = true;
    updateSimCampaignStatus();
    applyCampaignLatestWeights();
    syncCampaignWeightLockUi();
  }
}

function onSimCampaignListAck(msg) {
  if (!msg.ok) return;
  simCampaigns = msg.campaigns || [];
  fillSimCampaignSelect(simCampaigns, activeCampaign?.campaign_id || "");
}

function clearSubjectLoginLocal() {
  localStorage.removeItem(SUBJECT_LOGIN_KEY);
}

function updateSubjectBar() {
  if (!el.subjectBar) return;
  if (!activeSubject) {
    el.subjectBar.classList.add("hidden");
    return;
  }
  el.subjectBar.classList.remove("hidden");
  if (el.subjectBarId) el.subjectBarId.textContent = activeSubject;
  const sug = activeSubjectInfo?.suggest_session_id || "";
  if (el.subjectBarSession) {
    el.subjectBarSession.textContent = sug ? `建议 session ${sug}` : "";
  }
  const w = activeSubjectInfo?.current_weights;
  if (el.subjectBarWeights) {
    el.subjectBarWeights.textContent = w?.ok
      ? `权重 current (${w.release_pass === true ? "gate PASS" : w.release_pass === false ? "gate FAIL" : "—"})`
      : "权重：OpenBMI 底座";
  }
}

/** 当前协议板块：v1 / v2 / v3 / v4（仿真返回 sim） */
function currentProtocolBoard() {
  const pm = String(
    (el.form?.querySelector('input[name="phase_mode"]:checked') || {}).value || "",
  );
  if (pm === "v2_session") return "v2";
  if (pm === "v3_session") return "v3";
  if (pm === "v4_session") return "v4";
  if (pm === "sim_v3_session") return "sim";
  return "v1";
}

function phaseModeToBoard(phaseMode) {
  const pm = String(phaseMode || "").trim();
  if (pm === "v2_session") return "v2";
  if (pm === "v3_session") return "v3";
  if (pm === "v4_session") return "v4";
  if (pm === "sim_v3_session") return "sim";
  if (pm === "phase2_full" || pm === "phase1" || !pm) return "v1";
  return pm;
}

function boardLabel(board) {
  if (board === "v1") return "v1";
  if (board === "v2") return "v2";
  if (board === "v3") return "v3";
  if (board === "v4") return "v4";
  if (board === "sim") return "仿真";
  return board || "未知";
}

/** 规范化手输编号：`1`/`01`→`w01`，`w1`→`w01`；保留 ws/ses 前缀 */
function normalizeSessionIdInput(raw) {
  let s = String(raw || "").trim().toLowerCase();
  if (!s) return "";
  if (/^\d+$/.test(s)) return `w${s.padStart(2, "0")}`;
  const m = s.match(/^(w|ws|ses)(\d+)$/i);
  if (m) return `${m[1].toLowerCase()}${m[2].padStart(2, "0")}`;
  return s;
}

function readSetupSessionId() {
  return normalizeSessionIdInput(el.setupSessionId?.value || "");
}

/** 从会话列表按板块计算下一号 wNN（兼容历史 ws/ses） */
function computeSuggestSessionIdForBoard(board) {
  let maxN = 0;
  for (const s of subjectSessions()) {
    if (phaseModeToBoard(s.phase_mode) !== board) continue;
    const m = String(s.session_id || s.run_id || "").trim().match(/^(?:w|ws|ses)(\d+)$/i);
    if (m) maxN = Math.max(maxN, Number(m[1]));
  }
  return `w${String(maxN + 1).padStart(2, "0")}`;
}

/** 当前板块的建议会话编号（优先服务端 by_board；若已占用则本地重算） */
function suggestSessionIdForCurrentBoard() {
  const board = currentProtocolBoard();
  const by =
    activeSubjectInfo?.suggest_session_ids_by_board ||
    activeSubjectInfo?.index?.suggest_session_ids_by_board ||
    {};
  let sug = "";
  if (by && by[board]) sug = String(by[board]).trim();
  else if (board === "sim") {
    sug = String(activeSubjectInfo?.suggest_session_id || "run1").trim();
  } else {
    sug = computeSuggestSessionIdForBoard(board);
  }
  // 服务端建议可能过期（仍指向已有编号）→ 按本机会话列表重算
  if (board !== "sim" && sug && sessionsMatchingId(sug, { boardOnly: true }).length) {
    sug = computeSuggestSessionIdForBoard(board);
  }
  return sug;
}

function syncSuggestSessionIdForBoard() {
  if (!activeSubjectInfo || activeSimMode) return suggestSessionIdForCurrentBoard();
  const sug = suggestSessionIdForCurrentBoard();
  activeSubjectInfo.suggest_session_id = sug;
  if (!activeSubjectInfo.suggest_session_ids_by_board) {
    activeSubjectInfo.suggest_session_ids_by_board = {};
  }
  activeSubjectInfo.suggest_session_ids_by_board[currentProtocolBoard()] = sug;
  return sug;
}

/** 统一取会话列表：登录包在 index.sessions，info 包在顶层 sessions */
function subjectSessions(info = activeSubjectInfo) {
  if (!info) return [];
  if (Array.isArray(info.sessions) && info.sessions.length) return info.sessions;
  if (Array.isArray(info.index?.sessions)) return info.index.sessions;
  return Array.isArray(info.sessions) ? info.sessions : [];
}

/** 全部已有会话编号（去重），附带各编号出现过的板块标签 */
function listExistingSessionEntries() {
  const byId = new Map();
  for (const s of subjectSessions()) {
    const id = String(s.session_id || s.run_id || "").trim();
    if (!id) continue;
    const key = id.toLowerCase();
    const board = phaseModeToBoard(s.phase_mode);
    let ent = byId.get(key);
    if (!ent) {
      ent = { id, boards: [], count: 0 };
      byId.set(key, ent);
    }
    ent.count += 1;
    if (!ent.boards.includes(board)) ent.boards.push(board);
  }
  const out = [...byId.values()];
  out.sort((a, b) => {
    const ma = a.id.match(/^(w|ws|ses|run)(\d+)$/i);
    const mb = b.id.match(/^(w|ws|ses|run)(\d+)$/i);
    if (ma && mb) {
      return Number(ma[2]) - Number(mb[2]);
    }
    return a.id.localeCompare(b.id, undefined, { numeric: true });
  });
  return out;
}

/** @deprecated 保留兼容：默认返回全部编号；传 board 则只返回该板块 */
function uniqueExistingSessionIds({ board = null } = {}) {
  const entries = listExistingSessionEntries();
  if (!board) return entries.map((e) => e.id);
  return entries.filter((e) => e.boards.includes(board)).map((e) => e.id);
}

/** 按板块汇总已有会话编号（同板块多场同号合并计数） */
function listSessionIdsGroupedByBoard() {
  const boards = ["v2", "v3", "v4", "v1"];
  const map = Object.fromEntries(boards.map((b) => [b, new Map()]));
  for (const s of subjectSessions()) {
    const id = String(s.session_id || s.run_id || "").trim();
    if (!id) continue;
    const board = phaseModeToBoard(s.phase_mode);
    if (!map[board]) map[board] = new Map();
    const key = id.toLowerCase();
    const prev = map[board].get(key);
    if (prev) prev.count += 1;
    else map[board].set(key, { id, count: 1 });
  }
  const sortIds = (arr) => {
    arr.sort((a, b) => {
      const ma = a.id.match(/^(w|ws|ses|run)(\d+)$/i);
      const mb = b.id.match(/^(w|ws|ses|run)(\d+)$/i);
      if (ma && mb) return Number(ma[2]) - Number(mb[2]);
      return a.id.localeCompare(b.id, undefined, { numeric: true });
    });
    return arr;
  };
  return boards.map((board) => ({
    board,
    ids: sortIds([...(map[board]?.values() || [])]),
    next: suggestSessionIdForBoardKey(board),
  }));
}

function suggestSessionIdForBoardKey(board) {
  const by =
    activeSubjectInfo?.suggest_session_ids_by_board ||
    activeSubjectInfo?.index?.suggest_session_ids_by_board ||
    {};
  let sug = by && by[board] ? String(by[board]).trim() : "";
  if (!sug) return computeSuggestSessionIdForBoard(board);
  const want = normalizeSessionIdInput(sug);
  const taken = subjectSessions().some(
    (s) =>
      String(s.session_id || s.run_id || "").toLowerCase() === want &&
      phaseModeToBoard(s.phase_mode) === board,
  );
  return taken ? computeSuggestSessionIdForBoard(board) : sug;
}

function pickSessionIdFromMap(sessionId, { fromBoard = null } = {}) {
  if (activeSimMode) return;
  const sid = normalizeSessionIdInput(sessionId);
  if (!sid) return;
  setSessionIdSelectValue(sid, { overwrite: false });
  onSessionIdSelectChange();
  renderSessionBoardMap();
}

/** 可见对照：各板块已有编号 + 下一建议号（点击填入） */
function renderSessionBoardMap() {
  const root = el.sessionIdBoardMap;
  if (!root) return;
  if (activeSimMode) {
    root.classList.add("hidden");
    root.innerHTML = "";
    return;
  }
  root.classList.remove("hidden");
  const current = currentProtocolBoard();
  const selected = normalizeSessionIdInput(el.setupSessionId?.value || "");
  const groups = listSessionIdsGroupedByBoard().filter(
    (g) => g.ids.length > 0 || g.board === current || g.board === "v2" || g.board === "v3" || g.board === "v4",
  );

  root.innerHTML = "";
  const title = document.createElement("p");
  title.className = "session-board-map-title";
  title.textContent = "各板块已有会话（点击编号填入；虚线 = 该板块建议下一号）";
  root.appendChild(title);

  for (const g of groups) {
    // 无会话且非常用板块：跳过 v1 空行
    if (!g.ids.length && g.board === "v1" && current !== "v1") continue;

    const row = document.createElement("div");
    row.className = "session-board-row" + (g.board === current ? " is-current" : "");
    row.dataset.board = g.board;

    const name = document.createElement("span");
    name.className = "session-board-name";
    name.textContent = boardLabel(g.board);
    row.appendChild(name);

    const nextHint = document.createElement("span");
    nextHint.className = "session-board-next";
    nextHint.textContent = `下一号 ${g.next}`;
    row.appendChild(nextHint);

    const chips = document.createElement("div");
    chips.className = "session-board-chips";

    if (!g.ids.length) {
      const empty = document.createElement("span");
      empty.className = "session-id-chip is-empty";
      empty.textContent = "（尚无）";
      chips.appendChild(empty);
    } else {
      for (const ent of g.ids) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "session-id-chip";
        if (selected && ent.id.toLowerCase() === selected) btn.classList.add("is-selected");
        btn.textContent = ent.count > 1 ? `${ent.id}×${ent.count}` : ent.id;
        btn.title =
          g.board === current
            ? `填入 ${ent.id}（本板块已有，可能覆盖）`
            : `填入 ${ent.id}（其他板块已有同号时可并存）`;
        btn.addEventListener("click", () =>
          pickSessionIdFromMap(ent.id, { fromBoard: g.board }),
        );
        chips.appendChild(btn);
      }
    }

    const sugBtn = document.createElement("button");
    sugBtn.type = "button";
    sugBtn.className = "session-id-chip is-suggest";
    if (selected && g.next.toLowerCase() === selected) sugBtn.classList.add("is-selected");
    sugBtn.textContent = `新建 ${g.next}`;
    sugBtn.title = `填入 ${g.board} 建议下一号 ${g.next}`;
    sugBtn.addEventListener("click", () => {
      if (g.board === current) {
        applySuggestedSessionId({ force: true });
      } else {
        // 其他板块的「下一号」也可手选用（跨板块并存）
        pickSessionIdFromMap(g.next, { fromBoard: g.board });
      }
      renderSessionBoardMap();
    });
    chips.appendChild(sugBtn);

    row.appendChild(chips);
    root.appendChild(row);
  }
}

function setSessionIdSelectValue(sessionId, { overwrite = false } = {}) {
  if (!el.setupSessionId) return;
  const want = normalizeSessionIdInput(sessionId);
  if (!want) return;
  el.setupSessionId.value = want;
  if (el.sessionOverwrite) el.sessionOverwrite.checked = Boolean(overwrite);
  updateSessionIdConflictUi();
}

function formatSessionOptionLabel(entry, currentBoard) {
  const boards = (entry.boards || []).map(boardLabel).join("/");
  const n = entry.count > 1 ? `（${entry.count} 场）` : "";
  const cur = entry.boards.includes(currentBoard) ? "" : " · 其他板块";
  return `${entry.id}（${boards || "?"}${n}）${cur}`;
}

/** 填充会话编号 datalist：建议号 + 已有编号；输入框可手输任意号 */
function fillSessionIdSelect(preferredId) {
  if (!el.setupSessionId) return;
  if (activeSimMode) {
    renderSessionBoardMap();
    return;
  }
  const board = currentProtocolBoard();
  const bl = boardLabel(board);
  const sug = String(syncSuggestSessionIdForBoard() || "w01").trim();
  const entries = listExistingSessionEntries();
  const prev = normalizeSessionIdInput(preferredId || el.setupSessionId.value || sug);

  if (el.sessionIdList) {
    el.sessionIdList.innerHTML = "";
    const addOpt = (value, label) => {
      const opt = document.createElement("option");
      opt.value = value;
      if (label) opt.label = label;
      el.sessionIdList.appendChild(opt);
    };
    addOpt(sug, `新建建议 · ${sug}（${bl}）`);
    for (const ent of entries) {
      if (ent.id.toLowerCase() === sug.toLowerCase()) continue;
      addOpt(ent.id, formatSessionOptionLabel(ent, board));
    }
  }

  const allIds = entries.map((e) => e.id);
  const matchExisting = allIds.find((id) => id.toLowerCase() === prev.toLowerCase());
  el.setupSessionId.value = matchExisting || prev || sug;
  if (el.sessionOverwrite) {
    const boardHits = sessionsMatchingId(el.setupSessionId.value, { boardOnly: true });
    el.sessionOverwrite.checked =
      boardHits.length > 0 &&
      el.setupSessionId.value.toLowerCase() !== sug.toLowerCase() &&
      Boolean(el.sessionOverwrite.checked);
  }
  updateSessionIdConflictUi();
  renderSessionBoardMap();
}

/** 冲突检测：同编号默认查全部板块（覆盖会归档全部同号目录） */
function sessionsMatchingId(sessionId, { boardOnly = false } = {}) {
  const want = normalizeSessionIdInput(sessionId);
  if (!want) return [];
  const wantBoard = currentProtocolBoard();
  return subjectSessions().filter((s) => {
    if (String(s.session_id || s.run_id || "").toLowerCase() !== want) return false;
    if (!boardOnly) return true;
    return phaseModeToBoard(s.phase_mode) === wantBoard;
  });
}

/** 真机：检测会话编号冲突，提示建议编号与覆盖选项 */
function updateSessionIdConflictUi() {
  if (!el.sessionIdHint) return;
  if (activeSimMode) {
    el.sessionIdHint.textContent = "仿真模式：会话编号与 run 一致";
    el.sessionOverwriteWrap?.classList.add("hidden");
    if (el.sessionOverwrite) el.sessionOverwrite.checked = false;
    renderSessionBoardMap();
    return;
  }
  const board = currentProtocolBoard();
  const bl = boardLabel(board);
  const sid = normalizeSessionIdInput(el.setupSessionId?.value || "");
  const sug = String(activeSubjectInfo?.suggest_session_id || "").trim();
  const boardHits = sessionsMatchingId(sid, { boardOnly: true });
  const allHits = sessionsMatchingId(sid);
  if (!sid) {
    el.sessionIdHint.textContent = sug
      ? `当前 ${bl}：可手输编号（建议 ${sug}），或点「建议号」/下方虚线芯片`
      : `当前 ${bl}：请输入会话编号（如 w01）`;
    el.sessionOverwriteWrap?.classList.add("hidden");
    highlightSessionBoardMapSelection();
    return;
  }
  if (boardHits.length) {
    const hits = boardHits;
    const names = hits.map((s) => s.dir).slice(0, 3).join("、");
    const over = Boolean(el.sessionOverwrite?.checked);
    el.sessionIdHint.textContent = over
      ? `将覆盖 ${bl}「${sid}」（${hits.length} 场：${names || sid}）→ 旧目录归档后重采`
      : `本板块已有「${sid}」（${hits.length} 场）。勾选覆盖，或改输未用编号（建议 ${sug || "下一号"}）`;
    el.sessionOverwriteWrap?.classList.remove("hidden");
  } else if (allHits.length && !boardHits.length) {
    const boards = [
      ...new Set(allHits.map((s) => boardLabel(phaseModeToBoard(s.phase_mode)))),
    ];
    el.sessionIdHint.textContent =
      `当前 ${bl}：将新建 ${sid}（其他板块已有同号：${boards.join("/")}，可并存）`;
    el.sessionOverwriteWrap?.classList.add("hidden");
    if (el.sessionOverwrite) el.sessionOverwrite.checked = false;
  } else {
    const isSug = sug && sid.toLowerCase() === sug.toLowerCase();
    el.sessionIdHint.textContent = isSug
      ? `当前 ${bl}：将新建建议号 ${sid}`
      : `当前 ${bl}：将新建手输编号 ${sid}` + (sug ? `（系统建议 ${sug}）` : "");
    el.sessionOverwriteWrap?.classList.add("hidden");
    if (el.sessionOverwrite) el.sessionOverwrite.checked = false;
  }
  highlightSessionBoardMapSelection();
}

/** 仅更新芯片选中态（输入时不必整表重建） */
function highlightSessionBoardMapSelection() {
  const root = el.sessionIdBoardMap;
  if (!root || root.classList.contains("hidden")) return;
  const selected = normalizeSessionIdInput(el.setupSessionId?.value || "");
  root.querySelectorAll(".session-id-chip").forEach((btn) => {
    if (btn.classList.contains("is-empty")) return;
    const raw = String(btn.textContent || "").replace(/×\d+$/, "").replace(/^新建\s+/, "").trim();
    const id = normalizeSessionIdInput(raw);
    btn.classList.toggle("is-selected", Boolean(selected) && id === selected);
  });
  const current = currentProtocolBoard();
  root.querySelectorAll(".session-board-row").forEach((row) => {
    row.classList.toggle("is-current", row.dataset.board === current);
  });
}

function onSessionIdSelectChange() {
  if (activeSimMode) {
    updateSessionIdConflictUi();
    return;
  }
  const bl = boardLabel(currentProtocolBoard());
  const sid = readSetupSessionId();
  if (el.setupSessionId) el.setupSessionId.value = sid;
  const sug = String(syncSuggestSessionIdForBoard() || "").trim();
  const boardHits = sessionsMatchingId(sid, { boardOnly: true });

  if (boardHits.length) {
    const ok = confirm(
      `当前 ${bl} 已有会话编号「${sid}」（${boardHits.length} 场）。\n\n` +
        `确定：覆盖（旧目录移入 _archived）\n` +
        `取消：改回建议号 ${sug || "下一编号"}`,
    );
    if (ok) {
      if (el.sessionOverwrite) el.sessionOverwrite.checked = true;
      updateSessionIdConflictUi();
    } else {
      if (el.sessionOverwrite) el.sessionOverwrite.checked = false;
      setSessionIdSelectValue(sug || "w01", { overwrite: false });
    }
    return;
  }
  if (el.sessionOverwrite) el.sessionOverwrite.checked = false;
  updateSessionIdConflictUi();
}

function applySuggestedSessionId({ force = false } = {}) {
  if (activeSimMode) return;
  const sug = String(syncSuggestSessionIdForBoard() || "").trim();
  if (!sug) return;
  fillSessionIdSelect(force ? sug : undefined);
  setSessionIdSelectValue(sug, { overwrite: false });
}

function applySubjectToSetup(info) {
  if (!info) return;
  const sid = info.subject_id || info.subject?.subject_id;
  if (el.setupSubjectId) el.setupSubjectId.value = sid || "";
  // 归一化：登录包可能只有 index.sessions
  if (!Array.isArray(info.sessions) || !info.sessions.length) {
    info.sessions = info.index?.sessions || [];
  }
  if (activeSubjectInfo && activeSubjectInfo === info) {
    activeSubjectInfo.sessions = info.sessions;
  } else if (activeSubjectInfo?.subject_id === sid) {
    activeSubjectInfo.sessions = info.sessions;
    if (info.suggest_session_id) {
      activeSubjectInfo.suggest_session_id = info.suggest_session_id;
    }
  }
  if (info.suggest_session_ids_by_board) {
    if (!activeSubjectInfo) activeSubjectInfo = info;
    activeSubjectInfo.suggest_session_ids_by_board = {
      ...(activeSubjectInfo.suggest_session_ids_by_board || {}),
      ...info.suggest_session_ids_by_board,
    };
  } else if (info.index?.suggest_session_ids_by_board) {
    if (!activeSubjectInfo) activeSubjectInfo = info;
    activeSubjectInfo.suggest_session_ids_by_board = {
      ...(activeSubjectInfo.suggest_session_ids_by_board || {}),
      ...info.index.suggest_session_ids_by_board,
    };
  }
  activeSimMode = Boolean(info.sim_mode);
  if (info.sim_mode) {
    // 仿真：会话编号 = run_id，优先服务端 suggest（跑完 run3 → run4）
    refreshSimRunSuggestion();
  } else if (el.setupSessionId) {
    const sug = syncSuggestSessionIdForBoard();
    fillSessionIdSelect(sug);
    setSessionIdSelectValue(sug, { overwrite: false });
  }
  updateSessionIdConflictUi();
  const paths = info.sim_mode
    ? {
        save_root: info.sessions_dir || `experiment_game/data/sim_subjects/${sid}/sessions`,
        subject_root: info.subject_root,
      }
    : info.subject_root
      ? {
          save_root: `experiment_game/data/subjects/${sid}/sessions`,
          subject_root: info.subject_root,
        }
      : null;
  if (info.sim_mode) {
    simCampaigns = info.campaigns || [];
    fillSimCampaignSelect(simCampaigns);
    requestSimCatalog();
  }
  syncWorkModeUi();
  if (paths) {
    const saveRootInput = el.form?.querySelector("[name=save_root]");
    if (saveRootInput) saveRootInput.value = paths.save_root;
  }
  const cw = info.current_weights;
  if (cw?.ok && cw.task_ckpt && cw.three_ckpt && !activeCampaign) {
    if (el.s3TaskCkpt) el.s3TaskCkpt.value = cw.task_ckpt;
    if (el.s3ThreeCkpt) el.s3ThreeCkpt.value = cw.three_ckpt;
    if (el.modelPreset) {
      const hit = MODEL_PRESETS.find((p) => p.task === cw.task_ckpt && p.three === cw.three_ckpt);
      el.modelPreset.value = hit ? hit.id : "custom";
    }
    updateWeightsStatusFromInputs();
  } else if (cw?.ok === false) {
    applyModelPreset("openbmi_baseline");
  }
  updateSubjectBar();
  updateReplayAdvice("setup", { autoApply: false });
  if (activeCampaign) applyCampaignLatestWeights();
  syncCampaignWeightLockUi();
}

function renderFtSessionList(sessions) {
  if (!el.ftSessionList) return;
  el.ftSessionList.innerHTML = "";
  const autoSelectRoot = lastFtAutoSelectRoot;
  lastFtAutoSelectRoot = null;
  const autoNorm = autoSelectRoot ? normSessionPath(autoSelectRoot) : "";
  const list = sortFtSessionsForPanel(sessions, autoSelectRoot);
  ftSessionCatalog = list.slice();
  if (!list.length) {
    el.ftSessionList.textContent = "暂无历史 session";
    updateReplayAdvice("panel");
    updateFtRampHint();
    return;
  }
  for (const s of list) {
    const row = document.createElement("div");
    const isThisRun = autoNorm && normSessionPath(s.path) === autoNorm;
    // Leave-Next / 计划：留当前作评估，默认勾选之前所有合格 session
    const checkPrev =
      Boolean(s.ft_eligible) && !isThisRun && Boolean(el.ftLeaveNext?.checked !== false);
    row.className =
      "ft-session-row" +
      (s.electrode_ok ? "" : " warn") +
      (isThisRun ? " ft-session-current" : "");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checkPrev;
    cb.dataset.path = s.path || "";
    cb.dataset.runId = String(s.session_id || s.run_id || "").toLowerCase();
    cb.dataset.thisRun = isThisRun ? "1" : "0";
    cb.addEventListener("change", () => {
      replayAdviceManualOverride.panel = false;
      updateReplayAdvice("panel");
    });
    const meta = document.createElement("div");
    const wa = s.window_acc != null ? `${(s.window_acc * 100).toFixed(0)}%` : "—";
    const pa = s.primary_acc != null ? `${(s.primary_acc * 100).toFixed(0)}%` : "—";
    const nw = sessionWindowEstimate(s);
    const tag = isThisRun ? " · 本场(默认不入 FT)" : "";
    meta.textContent = `${s.dir}${s.record_excluded ? " · 未记入" : ""}${tag} · ~${nw} 窗 · 窗acc ${wa} · valid-primary ${pa}`;
    const warn = document.createElement("div");
    warn.className = "muted";
    warn.textContent = s.electrode_ok ? "电极 OK" : (s.electrode_warnings || []).join(" · ") || "电极 ⚠";
    row.appendChild(cb);
    row.appendChild(meta);
    row.appendChild(warn);
    el.ftSessionList.appendChild(row);
  }
  replayAdviceManualOverride.panel = false;
  applyLeaveNextFtSelection(autoNorm);
  updateReplayAdvice("panel");
  updateFtRampHint();
}

/** Leave-Next：勾选当前场之前的 session，排除本场。仿真走 Campaign queue；真机走目录序。 */
function applyLeaveNextFtSelection(evalSessionPathNorm) {
  if (!el.ftLeaveNext?.checked) return;

  // 仿真 Campaign：按 queue 切 train / eval
  if (activeSimMode && activeCampaign) {
    const queue = (activeCampaign.session_queue || []).map((r) => String(r).toLowerCase());
    const done = {};
    for (const item of activeCampaign.sessions_completed || []) {
      const rid = String(item.run_id || "").toLowerCase();
      if (rid && item.session_dir) done[rid] = normSessionPath(item.session_dir);
    }
    let evalRun = resolveFtEvalRunId(evalSessionPathNorm);
    if (!evalRun || !queue.includes(evalRun)) return;
    const idx = queue.indexOf(evalRun);
    const trainRuns = new Set(queue.slice(0, idx));
    const trainPaths = new Set(
      [...trainRuns].map((rid) => done[rid]).filter(Boolean),
    );
    el.ftSessionList?.querySelectorAll("input[type=checkbox]").forEach((cb) => {
      const p = normSessionPath(cb.dataset.path || "");
      cb.checked = trainPaths.has(p);
    });
    return;
  }

  // 真机 / 无 Campaign：再确认排除本场，勾选其余合格项
  const evalNorm =
    evalSessionPathNorm ||
    (sessionRoot ? normSessionPath(sessionRoot) : "");
  el.ftSessionList?.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    const p = normSessionPath(cb.dataset.path || "");
    const isThis = evalNorm && p === evalNorm;
    const hit = ftSessionCatalog.find((s) => normSessionPath(s.path) === p);
    cb.checked = !isThis && Boolean(hit?.ft_eligible);
  });
}

function resolveFtEvalRunId(evalSessionPathNorm) {
  const fromRoot = evalSessionPathNorm || (lastFtAutoSelectRoot ? normSessionPath(lastFtAutoSelectRoot) : "");
  if (fromRoot && activeSubjectInfo?.sessions) {
    const hit = (activeSubjectInfo.sessions || []).find(
      (s) => normSessionPath(s.path) === fromRoot,
    );
    const sid = String(hit?.session_id || hit?.run_id || "").toLowerCase();
    if (sid.startsWith("run")) return sid;
  }
  if (sessionRoot && activeSubjectInfo?.sessions) {
    const hit = (activeSubjectInfo.sessions || []).find(
      (s) => normSessionPath(s.path) === normSessionPath(sessionRoot),
    );
    const sid = String(hit?.session_id || hit?.run_id || "").toLowerCase();
    if (sid.startsWith("run")) return sid;
  }
  const runInp = el.form?.querySelector('[name="sim_run_id"]');
  const v = String(runInp?.value || el.setupSessionId?.value || "").trim().toLowerCase();
  if (v.startsWith("run")) return v;
  return null;
}

function updateFtRampHint() {
  if (!el.ftRampHint) return;
  if (!el.ftLeaveNext?.checked) {
    el.ftRampHint.textContent = "Leave-Next 关：可手动勾选本场与历史 session";
    return;
  }
  if (!activeSimMode || !activeCampaign) {
    const n = el.ftSessionList
      ? [...el.ftSessionList.querySelectorAll("input[type=checkbox]:checked")].length
      : 0;
    el.ftRampHint.textContent =
      `Leave-Next：本场不入 FT · 已勾选历史 ${n} 场（可手动改）`;
    return;
  }
  const evalRun = resolveFtEvalRunId();
  const queue = (activeCampaign.session_queue || []).map((r) => String(r).toLowerCase());
  if (!evalRun || !queue.includes(evalRun)) {
    el.ftRampHint.textContent = "Leave-Next：等待确定 eval run…";
    return;
  }
  const rStage = queue.indexOf(evalRun);
  const train = queue.slice(0, rStage);
  const replayOn = rStage >= 1 && rStage < 4;
  el.ftRampHint.textContent =
    `R${rStage} · eval=${evalRun} · FT train=[${train.join(", ") || "—"}]` +
    (rStage === 0
      ? " · R0 不 FT（仅底座评估）"
      : ` · replay 建议 ${replayOn ? "开 0.10" : "关"}`);
  if (activeCampaign.manifest_path) {
    send({
      type: "ramp_status",
      subject_id: activeSubject,
      campaign_manifest: activeCampaign.manifest_path,
      eval_run_id: evalRun,
    });
  }
}

function collectSelectedFtSessions() {
  const paths = new Set(collectFtSessionPaths());
  return ftSessionCatalog.filter((s) => paths.has(s.path));
}

function resetFtPanel() {
  lastFtRunDir = null;
  lastFtGatePass = null;
  ftBusy = false;
  if (el.ftResult) {
    el.ftResult.classList.add("hidden");
    el.ftResult.textContent = "";
  }
  if (el.ftStatus) el.ftStatus.textContent = "";
  if (el.btnFtStart) {
    el.btnFtStart.disabled = false;
    el.btnFtStart.textContent = "开始微调";
  }
  if (el.btnFtPromote) el.btnFtPromote.classList.add("hidden");
  if (el.btnFtKeep) el.btnFtKeep.classList.add("hidden");
  sessionRecordExcluded = false;
  updateSessionNoRecordButton();
  const skip = document.querySelector('input[name="ft_mode"][value="skip"]');
  if (skip) skip.checked = true;
  if (el.ftExcludeInvalid) el.ftExcludeInvalid.checked = false;
  replayAdviceManualOverride.panel = false;
  applyFtReplayDefaults(lockedConfig?.experiment?.ft_defaults);
  applyFtAdvancedDefaults(lockedConfig?.experiment?.ft_defaults);
  updateReplayAdvice("panel");
}

function updateSessionNoRecordButton() {
  if (!el.btnSessionNoRecord) return;
  const show = Boolean(sessionRoot);
  el.btnSessionNoRecord.classList.toggle("hidden", !show);
  el.btnSessionNoRecord.disabled = sessionRecordExcluded || !sessionRoot;
  el.btnSessionNoRecord.textContent = sessionRecordExcluded
    ? "已标记：不记入记录"
    : "不记入实验记录";
}

function showFtPanel(msg) {
  if (!el.ftPanel) return;
  el.ftPanel.classList.remove("hidden");
  resetFtPanel();
  lastFtAutoSelectRoot = msg?.root || sessionRoot || null;
  const cw = activeSubjectInfo?.current_weights;
  if (el.ftCurrentWeights) {
    el.ftCurrentWeights.textContent = cw?.ok
      ? `当前 current：${cw.path || "—"}`
      : "当前无被试 current 权重（将使用 OpenBMI 底座）";
  }
  if (!msg?.train_eligible && !msg?.acq_enabled) {
    if (el.ftStatus) el.ftStatus.textContent = "本次未采集 EEG，不可微调";
    if (el.btnFtStart) el.btnFtStart.disabled = true;
  }
  sessionRecordExcluded = Boolean(msg?.record_excluded);
  lastCampaignManifest = msg?.campaign?.manifest || activeCampaign || null;
  updateSessionNoRecordButton();
  send({ type: "subject_info", subject_id: activeSubject || msg?.subject_id });
}

function collectFtSessionPaths() {
  const paths = [];
  if (!el.ftSessionList) return paths;
  el.ftSessionList.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    if (cb.checked && cb.dataset.path) paths.push(cb.dataset.path);
  });
  return paths;
}

function showView(name) {
  if (el.login) el.login.classList.toggle("hidden", name !== "login");
  el.setup.classList.toggle("hidden", name !== "setup");
  el.run.classList.toggle("hidden", name !== "run");
  el.summary.classList.toggle("hidden", name !== "summary");
  if (name !== "login") location.hash = name;
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
    el.v3PowerBars.innerHTML = v3BaselineWaitHtml();
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
  resetSessionScore(null);
  resetSessionVoteHistory();
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
  const sim = isSimV3Mode();
  const useCampaignQueue = Boolean(el.form.querySelector('[name="sim_use_campaign_queue"]')?.checked);
  const simRunId = sim ? resolveSimRunId(useCampaignQueue) : "";
  const replayAlign =
    String(
      (el.form.querySelector('input[name="sim_replay_align"]:checked') || {}).value || "schedule_align",
    ).trim() || "schedule_align";
  const board = sim
    ? "bci2a_replay"
    : fd.get("board_mode") ||
      (el.form.querySelector('input[name="board_mode"]:checked') || {}).value ||
      "synthetic";
  const acqEnabled = sim ? true : el.form.querySelector('[name="acq_enabled"]').checked;
  const layout = fd.get("save_layout") || "phase_folders";
  return {
    schema_version: 2,
    subject: {
      subject_id: String(fd.get("subject_id") || "").trim(),
      session_id: sim
        ? (simRunId || resolveSimRunId(false))
        : normalizeSessionIdInput(fd.get("session_id") || ""),
      notes: String(fd.get("notes") || ""),
    },
    acquisition: {
      enabled: acqEnabled,
      board_mode: board,
      serial_port: String(fd.get("serial_port") || "COM5").trim(),
      sample_rate_hz: 250,
      markers_lsl: acqEnabled,
      filter: {
        enabled: el.form.querySelector('[name="filter_enabled"]')?.checked === true,
        bandpass_low_hz: Number(fd.get("bandpass_low_hz") || 0.5),
        bandpass_high_hz: Number(fd.get("bandpass_high_hz") || 45),
        notch_enabled: el.form.querySelector('[name="notch_enabled"]')?.checked === true,
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
      open_subject_page: sim
        ? true
        : isV4Mode()
          ? false
          : el.form.querySelector('[name="open_subject_page"]').checked,
      skip_adapt: el.form.querySelector('[name="skip_adapt"]')?.checked || false,
      skip_learn: el.form.querySelector('[name="skip_learn"]')?.checked || false,
      skip_gate: el.form.querySelector('[name="skip_gate"]')?.checked || false,
      protocol_locked: el.form.querySelector('[name="protocol_locked"]')?.checked !== false,
      overwrite_session_id: Boolean(el.sessionOverwrite?.checked),
      ft_defaults: { ...readFtReplayOptions(el.form), ...readFtAdvancedOptions() },
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
      subject_feedback_mode: sim
        ? "arm_reach"
        : String(fd.get("subject_feedback_mode") || "none").trim() === "arm_reach"
          ? "arm_reach"
          : "none",
    },
    extensions: sim
      ? {
          sim: {
            run_id: useCampaignQueue
              ? ""
              : (simRunId || resolveSimRunId(false)),
            session_trials_total: Number(fd.get("sim_trials_total") || 36),
            include_rest: true,
            replay_speed: Number(fd.get("sim_replay_speed") || 4),
            replay_align: replayAlign,
            use_campaign_queue: useCampaignQueue,
            campaign_manifest: activeCampaign?.manifest_path || null,
          },
        }
      : {},
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
  if (!activeSimMode) {
    fillSessionIdSelect(cfg.subject?.session_id);
    if (cfg.subject?.session_id) {
      setSessionIdSelectValue(cfg.subject.session_id, {
        overwrite: Boolean(cfg.experiment?.overwrite_session_id),
      });
    }
  } else {
    set("session_id", cfg.subject?.session_id);
  }
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
  applyV3OverridesToForm(cfg.experiment?.v3_overrides || {});
  applyFtReplayDefaults(cfg.experiment?.ft_defaults);
  applyFtAdvancedDefaults(cfg.experiment?.ft_defaults);
  // v3 overrides 里的权重也回填（与 v2 共用表单字段）
  const wOv = {
    ...(cfg.experiment?.v2_overrides || {}),
    ...(cfg.experiment?.v3_overrides || {}),
  };
  if (wOv.s3_task_ckpt && el.s3TaskCkpt) el.s3TaskCkpt.value = wOv.s3_task_ckpt;
  if (wOv.s3_three_ckpt && el.s3ThreeCkpt) el.s3ThreeCkpt.value = wOv.s3_three_ckpt;
  updateWeightsStatusFromInputs();
  const pm = cfg.experiment?.phase_mode || "phase2_full";
  for (const r of el.form.querySelectorAll('input[name="phase_mode"]')) {
    r.checked = r.value === pm;
  }
  ensureExperimentPhaseMode();
  syncWorkModeUi();
  syncProtocolLockUi();
  set("save_root", cfg.storage?.save_root);
  set("save_layout", cfg.storage?.save_layout || "phase_folders");
  set("auto_phase4", cfg.storage?.auto_phase4);
  set("remember_last_config", cfg.ui?.remember_last_config !== false);
  set("subject_feedback_mode", cfg.ui?.subject_feedback_mode || "none");
  set("skip_setup_if_unchanged", cfg.ui?.skip_setup_if_unchanged);
  set("operator_hotkeys", cfg.ui?.operator_hotkeys !== false);
  const filt = cfg.acquisition?.filter || {};
  set("filter_enabled", filt.enabled === true);
  set("bandpass_low_hz", filt.bandpass_low_hz ?? 0.5);
  set("bandpass_high_hz", filt.bandpass_high_hz ?? 45);
  set("notch_enabled", filt.notch_enabled === true);
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
  updateReplayAdvice("setup", { autoApply: false });
}

function syncAcqUi() {
  const simSession = Boolean(activeSimMode);
  const acqOn = simSession ? true : el.form.querySelector('[name="acq_enabled"]').checked;
  const cyton = el.form.querySelector('input[name="board_mode"]:checked')?.value === "cyton";
  if (el.acqWarn) el.acqWarn.classList.toggle("hidden", acqOn || simSession);
  if (el.guiHint) el.guiHint.classList.toggle("hidden", !cyton || simSession);
  if (el.deviceFs) el.deviceFs.disabled = !acqOn || !cyton || simSession;
  const layout = el.form.querySelector('[name="save_layout"]')?.value || "flat";
  if (el.saveHint) {
    if (simSession) {
      el.saveHint.textContent =
        "仿真回放：mat → RingBuffer → eeg.csv + alignment/（固定开启，无需采集开关）";
    } else {
      el.saveHint.textContent = acqOn
        ? layout === "phase_folders"
          ? "将写入 continuous/ + by_phase/ + alignment/（EEG 与 Marker 同一 LSL 时钟）"
          : "扁平落盘：会话根 eeg.csv + events.jsonl + session.meta.json；并写 alignment/"
        : "仅 events + meta，无脑电，不能 Phase4 训练";
    }
  }
}

function showErrors(list) {
  const busyHint = document.getElementById("setup-busy-hint");
  if (!list || !list.length) {
    el.errors.classList.add("hidden");
    el.errors.textContent = "";
    busyHint?.classList.add("hidden");
    return;
  }
  el.errors.classList.remove("hidden");
  el.errors.textContent = list.join("\n");
  const busy = list.some((x) => String(x).includes("已有会话在进行"));
  busyHint?.classList.toggle("hidden", !busy);
}

function requestAbortSession({ fromSetup = false } = {}) {
  const msg = fromSetup
    ? "确认中止后台未结束的会话？已写入数据将尽量保留。"
    : "确认中止本场实验？已写入数据将尽量保留。";
  if (!confirm(msg)) return;
  starting = false;
  if (fromSetup) {
    showErrors(["正在中止后台会话…（完成后可再点「开始实验」）"]);
  } else {
    showRunAlert("正在中止会话…", "abort");
  }
  send({ type: "operator", action: "abort" });
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
  // 本场锁定的时序构成（v2/v3：Cue前 Rest → prep → Cue → MI → ITI）
  if (msg.timing) {
    const isV23 = msg.phase_mode === "v2_session" || msg.phase_mode === "v3_session";
    renderTimeline(el.runTimeline, msg.timing, isV23 ? V23_TIMELINE_SEGMENTS : undefined);
    if (el.runTimingHint) {
      if (isV23) {
        const rest = msg.timing.rest_s ?? msg.timing.inter_trial_rest_s ?? 0;
        const prep = msg.timing.fixation_s ?? msg.timing.prep_s ?? 0;
        const cue = Number(msg.timing.cue_s) || 0;
        const iti = msg.timing.transition_s ?? msg.timing.iti_s ?? 0;
        const cueTxt = cue <= 0 ? "Cue(=MI onset)" : `cue ${cue}s`;
        el.runTimingHint.textContent =
          `Cue前 Rest ${rest}s → prep ${prep}s → ${cueTxt} → ` +
          `MI ${msg.timing.mi_s}s → ITI ${iti}s` +
          (msg.trial_total_s != null ? ` · 合计 ${msg.trial_total_s}s` : "");
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
    try {
      handleMessage(msg);
    } catch (err) {
      console.error("[operator] handleMessage", msg?.type, err);
    }
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
    fillModelPresets(msg.model_presets || [], msg.active_weights || null);
    // 优先：服务端文件默认 > 浏览器 localStorage > 内置
    applyConfigToForm(defaultsFromServer || local || builtinDefaults);
    if (msg.active_weights && !(defaultsFromServer?.experiment?.v3_overrides?.s3_task_ckpt)) {
      fillModelPresets(msg.model_presets || MODEL_PRESETS, msg.active_weights);
    }
    fillSerialPorts(msg.serial_ports || []);
    maybeShowReuseBar(defaultsFromServer || local);
    if (msg.defaults_warning) showErrors([msg.defaults_warning]);
    if (msg.active_subject && msg.active_subject_info) {
      activeSubject = msg.active_subject;
      activeSubjectInfo = msg.active_subject_info;
      if (!Array.isArray(activeSubjectInfo.sessions) || !activeSubjectInfo.sessions.length) {
        activeSubjectInfo.sessions = activeSubjectInfo.index?.sessions || [];
      }
      saveSubjectLoginLocal(msg.active_subject_info);
      applySubjectToSetup(msg.active_subject_info);
      send({ type: "subject_info", subject_id: activeSubject });
      showView("setup");
    } else {
      const saved = loadSubjectLoginLocal();
      if (saved?.subject_id) {
        send({
          type: "subject_login",
          subject_id: saved.subject_id,
          display_name: saved.display_name || "",
          sim_mode: Boolean(saved.sim_mode),
        });
      } else {
        showView("login");
        if (el.loginLast && saved) {
          el.loginLast.textContent = `上次：${saved.subject_id}`;
        }
      }
    }
  } else if (t === "subject_login_ack") {
    if (!msg.ok) {
      if (el.loginError) {
        el.loginError.textContent = msg.message || "登录失败";
        el.loginError.classList.remove("hidden");
      }
      return;
    }
    if (el.loginError) el.loginError.classList.add("hidden");
    activeSubject = msg.subject_id;
    activeSubjectInfo = msg;
    // 兼容旧后端：sessions 可能只在 index 里
    if (!Array.isArray(activeSubjectInfo.sessions) || !activeSubjectInfo.sessions.length) {
      activeSubjectInfo.sessions = activeSubjectInfo.index?.sessions || [];
    }
    activeSimMode = Boolean(msg.sim_mode);
    saveSubjectLoginLocal(msg);
    if (msg.sim_mode) {
      const simRadio = el.form?.querySelector('input[name="phase_mode"][value="sim_v3_session"]');
      if (simRadio) simRadio.checked = true;
    } else {
      ensureExperimentPhaseMode();
    }
    syncWorkModeUi();
    fillModelPresets(msg.model_presets || MODEL_PRESETS, msg.active_weights || null);
    applySubjectToSetup(msg);
    if (msg.sim_mode) requestSimCatalog();
    // 再拉一次完整 subject_info，确保会话列表最新
    send({ type: "subject_info", subject_id: activeSubject });
    showView("setup");
  } else if (t === "subject_logout_ack") {
    activeSubject = null;
    activeSubjectInfo = null;
    activeSimMode = false;
    clearSubjectLoginLocal();
    updateSubjectBar();
    syncWorkModeUi();
    showView("login");
  } else if (t === "subject_info_ack") {
    if (!msg.ok) return;
    if (msg.subject_id === activeSubject) {
      activeSubjectInfo = {
        ...activeSubjectInfo,
        ...msg,
        sessions: msg.sessions || msg.index?.sessions || activeSubjectInfo?.sessions || [],
        suggest_session_id:
          msg.suggest_session_id ||
          msg.index?.suggest_session_id ||
          activeSubjectInfo?.suggest_session_id,
      };
      if (msg.sim_mode) {
        simCampaigns = msg.campaigns || simCampaigns;
        fillSimCampaignSelect(simCampaigns, activeCampaign?.campaign_id || "");
        refreshSimRunSuggestion();
      }
      renderFtSessionList(msg.sessions);
      if (!activeSimMode) {
        const cur = String(el.setupSessionId?.value || "").trim();
        fillSessionIdSelect(cur || msg.suggest_session_id);
      }
      updateReplayAdvice("setup", { autoApply: false });
      updateSubjectBar();
    }
  } else if (t === "sim_catalog_ack") {
    onSimCatalogAck(msg);
  } else if (t === "sim_campaign_ack") {
    onSimCampaignAck(msg);
  } else if (t === "sim_campaign_list_ack") {
    onSimCampaignListAck(msg);
  } else if (t === "ramp_status_ack") {
    if (!msg.ok || !el.ftRampHint) return;
    const train = (msg.leave_next_train || []).map((x) => x.run_id).join(", ") || "—";
    const rec = msg.ft_replay_recommendation || {};
    el.ftRampHint.textContent =
      `R${msg.ramp_stage} · eval=${msg.eval_run_id || "—"} · FT train=[${train}]` +
      ` · ${rec.reason || ""}`;
    if (el.ftLeaveNext?.checked && rec.use_replay != null) {
      if (el.ftUseReplay) el.ftUseReplay.checked = Boolean(rec.use_replay);
      if (el.ftReplayRatio && rec.replay_ratio != null) {
        el.ftReplayRatio.value = String(rec.replay_ratio);
      }
      syncFtReplayRatioUi();
    }
  } else if (t === "finetune_ack") {
    if (!msg.ok && el.ftStatus) el.ftStatus.textContent = msg.message || "微调失败";
  } else if (t === "finetune_progress") {
    if (el.ftStatus) el.ftStatus.textContent = `微调中… ${msg.out_dir || ""}`;
    ftBusy = true;
    if (el.btnFtStart) {
      el.btnFtStart.disabled = true;
      el.btnFtStart.textContent = "微调中…";
    }
  } else if (t === "finetune_done") {
    ftBusy = false;
    if (el.btnFtStart) {
      el.btnFtStart.disabled = false;
      el.btnFtStart.textContent = "开始微调";
    }
    if (!msg.ok) {
      if (el.ftStatus) el.ftStatus.textContent = msg.message || "微调失败";
      return;
    }
    lastFtRunDir = msg.out_dir;
    const gate = msg.release_gate || {};
    const pass = msg.release_pass;
    lastFtGatePass = Boolean(pass);
    const ftRec = msg.three_ft || {};
    const esLine = msg.early_stop
      ? `早停 · best@${ftRec.best_epoch ?? "—"}/${msg.max_epochs ?? 20} epoch · patience ${msg.patience ?? 5}`
      : `固定 ${msg.fixed_epochs ?? ftRec.epochs_run ?? 5} epoch`;
    const detLine = msg.deterministic ? `seed ${msg.seed ?? 42}` : "非确定性";
    if (el.ftResult) {
      el.ftResult.classList.remove("hidden");
      const repLine = msg.use_replay
        ? `replay T0 @ ${((msg.replay_ratio ?? 0.1) * 100).toFixed(0)}%`
        : "replay 关闭（纯被试窗）";
      const failBits = Object.entries(gate.checks || {})
        .filter(([, ok]) => !ok)
        .map(([k]) => k);
      const saved = msg.weights_saved !== false;
      const scopeLine = msg.ft_scope ? `范围 ${msg.ft_scope}` : "范围 —";
      const gateLine = pass
        ? `门控（参考）：PASS · pred ${JSON.stringify(gate.pred_labels || {})}`
        : `门控（参考）：FAIL${failBits.length ? ` · ${failBits.join(", ")}` : ""} · pred ${JSON.stringify(gate.pred_labels || {})}${saved ? " · 权重已写入预设" : " · 无权重文件"}`;
      const promoteLine = msg.auto_promoted
        ? msg.force_promoted
          ? "已自动强制晋升 current（告警已落盘 force_promote_warning.json）"
          : "已自动晋升 current"
        : "未自动晋升";
      el.ftResult.innerHTML = [
        `<div>${scopeLine} · ${repLine} · ${esLine} · ${detLine}</div>`,
        `<div>three heldout <strong>${(msg.three_heldout * 100).toFixed(1)}%</strong> · task <strong>${(msg.task_heldout * 100).toFixed(1)}%</strong>（参考）</div>`,
        `<div>${gateLine}</div>`,
        `<div>${promoteLine}</div>`,
        `<div class="muted">已保存 ${msg.out_dir}</div>`,
      ].join("");
    }
    if (el.ftStatus) {
      if (msg.weights_saved === false) {
        el.ftStatus.textContent =
          "微调结束但未写出 best_*.pt（请重启服务后重跑；旧版本门控 FAIL 会删权重）";
      } else if (msg.auto_promoted) {
        el.ftStatus.textContent = msg.force_promoted
          ? "微调完成；门控 FAIL，已强制晋升 current（告警已落盘）"
          : "微调完成；门控 PASS，已自动晋升 current";
      } else {
        el.ftStatus.textContent = pass
          ? "微调完成；门控 PASS，可替换 current；预设列表已刷新"
          : "微调完成；门控 FAIL，权重已进预设，强制替换请确认";
      }
    }
    if (el.btnFtPromote) {
      el.btnFtPromote.classList.toggle("hidden", msg.weights_saved === false);
    }
    if (el.btnFtKeep) el.btnFtKeep.classList.remove("hidden");
    refreshModelPresetsFromServer(msg);
  } else if (t === "finetune_promote_ack") {
    if (msg.ok) {
      if (el.ftStatus) el.ftStatus.textContent = `已替换 current：${msg.current_dir || ""}`;
      refreshModelPresetsFromServer(msg);
      activeSubjectInfo = { ...activeSubjectInfo, current_weights: msg.weights };
      applySubjectToSetup(activeSubjectInfo);
      if (el.btnFtPromote) el.btnFtPromote.classList.add("hidden");
    } else if (el.ftStatus) {
      el.ftStatus.textContent = msg.message || "晋升失败";
    }
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
      if (activeCampaign?.campaign_id) {
        const pid = el.modelPreset?.value;
        if (pid && isZeroSamplePresetId(pid)) {
          campaignModelPicks[activeCampaign.campaign_id] = pid;
        }
      }
      syncCampaignWeightLockUi();
      if (el.reuseBar) el.reuseBar.classList.add("hidden");
      showView("run");
      if (!isV4Mode()) el.popupWarn.classList.remove("hidden");
      // v4 也可打开被试页展示 HUD；是否打开跟 Setup「打开诱导页」勾选一致
      tryOpenSubject();
    }
  } else if (t === "session_started") {
    if (msg.subject_url) subjectUrl = msg.subject_url;
    sessionRoot = msg.session_root || "";
    const v2 = msg.phase_mode === "v2_session";
    const v3 = msg.phase_mode === "v3_session" || msg.phase_mode === "sim_v3_session";
    const v4 = msg.phase_mode === "v4_session";
    const sim = msg.phase_mode === "sim_v3_session";
    setV2RunPanel(v2);
    setV3RunPanel(v3);
    setV4RunPanel(v4);
    setWeightsDisplay(msg.weights);
    if (v4) {
      showRunAlert("");
      onV4Start(msg);
      if (el.stPhase) el.stPhase.textContent = "v4";
      if (el.stStage) el.stStage.textContent = "质量检测";
    } else if (v3) {
      showRunAlert(sim ? `仿真 · ${msg.sim?.source_run || ""} · 回放` : "");
      setPhaseStepV3(sim ? "block" : "baseline");
      if (sim && msg.sim?.skip_session_baseline) {
        V3_EEG_STATE.baselineReady = false;
        V3_EEG_STATE.baselineMu = [];
        V3_EEG_STATE.baselineBeta = [];
        V3_POWER_ROWS = null;
        if (el.v3PowerBars) {
          el.v3PowerBars.innerHTML =
            `<div class="v3-pbar-wait">仿真：等待试次间 Rest 建立 ERD 基线（跳过块前 30s）</div>`;
        }
      }
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
    if (v3) {
      const blocks = msg.v3_config_effective?.blocks ?? 2;
      const tpb = msg.v3_config_effective?.trials_per_block ?? 18;
      const nMi = Number(blocks) * Number(tpb);
      const restS =
        msg.timing?.inter_trial_rest_s ??
        msg.v3_config_effective?.inter_trial_rest_s ??
        4;
      const v3Max =
        msg.session_score_max ?? openbmiSessionScoreMax(nMi, restS);
      resetSessionScore(v3Max, { nMi, interTrialRestS: restS });
    } else if (v2) {
      resetSessionScore(msg.session_score_max ?? null);
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
    const r = msg.report || {};
    const ov = r.overall || {};
    renderSummaryWindowAcc({
      window_acc: ov.acc_window,
      window_acc_n: ov.n_windows,
      report: r,
    });
    if (el.v3SummaryDetail) {
      el.v3SummaryDetail.classList.remove("hidden");
      const waTxt = formatWindowAccPct(ov.acc_window);
      const parts = [
        `v3 报告 · 质量=${r.quality_tier || "—"}`,
        `frozen=${r.frozen}`,
      ];
      if (waTxt) {
        const n = ov.n_windows;
        parts.push(n != null ? `窗级识别率 ${waTxt}（${n} 窗）` : `窗级识别率 ${waTxt}`);
      }
      el.v3SummaryDetail.textContent = parts.join(" · ");
    }
  } else if (t === "eeg_stale") {
    const age = msg.age_s != null ? Number(msg.age_s).toFixed(1) : "?";
    const text =
      msg.message ||
      `EEG 断流：已 ${age}s 无新样本。请检查 dongle / COM / USB，会话将中止。`;
    showRunAlert(text, "abort");
    playAlertBeep("alert");
    if (el.stAcq) el.stAcq.textContent = `断流 · ${age}s`;
    showErrors([text]);
  } else if (t === "v3_abort") {
    const reason = msg.reason || "operator_abort";
    const text = msg.eeg_stale || String(reason).startsWith("eeg_stale")
      ? `v3 已中止：EEG 断流（${reason}）。请检查采集盒后重开会话。`
      : `v3 已中止：${reason}`;
    showRunAlert(text, "abort");
    stopGuidanceCountdown();
    if (msg.eeg_stale || String(reason).startsWith("eeg_stale")) {
      showErrors([text]);
    }
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
    showRunAlert(`v2 中止：${kind}`, "abort");
    playAlertBeep("alert");
    stopGuidanceCountdown();
    if (el.v2StageHint) el.v2StageHint.textContent = `v2 已中止 · ${reason}`;
  } else if (t === "v2_gate") {
    updateV2Gate(msg);
  } else if (t === "v2_stage") {
    const v3Active = !el.v3Panel?.classList.contains("hidden");
    updateLiveJudgePanels(msg);
    if (v3Active) {
      markV3StageEvent(msg.stage);
      el.stPhase.textContent = msg.progress?.phase_step || msg.stage || "—";
      el.stStage.textContent = msg.stage || "—";
      if (msg.ctx?.trial_id != null) el.stTrial.textContent = msg.ctx.trial_id;
      if (msg.progress?.phase_step) setPhaseStepV3(msg.progress.phase_step);
      applySessionScore(msg);
      if (msg.stage === "trial_end" && msg.data?.summary?.score != null) {
        const n = Number(msg.data.summary.score);
        const txt = Number.isInteger(n) ? String(n) : String(Math.round(n));
        if (el.stV2Score) el.stV2Score.textContent = txt;
      }
      if (msg.stage === "guidance_begin") {
        el.btnV3Guidance?.classList.remove("hidden");
        if (el.btnV3Guidance) el.btnV3Guidance.disabled = false;
        if (msg.data?.auto) {
          showRunAlert("合成/仿真：动觉引导自动确认", "ok");
          stopGuidanceCountdown();
          el.btnV3Guidance?.classList.add("hidden");
        } else {
          showRunAlert("动觉引导中 — 请完成双手分别抓握杯子后点「确认动觉引导完成」", "gate-pass");
          startGuidanceCountdown(msg.data?.timeout_s ?? 600);
        }
      }
      if (msg.stage === "guidance_end") {
        el.btnV3Guidance?.classList.add("hidden");
        stopGuidanceCountdown();
        if (msg.data?.passed) showRunAlert("");
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
      updateV2Progress(msg.progress, msg.score, msg);
      applySessionScore(msg);
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
        showRunAlert("准入建议：已过线 · 请操作员按 G /「准入」确认后进入游戏", "gate-pass");
        playAlertBeep("guidance");
        setPhaseStepV2("gate");
      }
      if (msg.stage === "weak_mi") {
        showRunAlert("准入建议：weak_mi · 请操作员确认是否仍进入游戏（Esc 可中止）", "degraded");
        el.v2WeakMi?.classList.remove("hidden");
        setPhaseStepV2("gate");
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
  } else if (t === "session_finishing") {
    // 试次跑完、落盘前：先切摘要页，避免对齐静默期断线后卡在运行页
    starting = false;
    showView("summary");
    if (msg.root) {
      sessionRoot = msg.root;
      if (el.summaryRoot) el.summaryRoot.textContent = sessionRoot;
    }
    if (el.summaryMsg) {
      el.summaryMsg.textContent = msg.message || "会话试次已结束，正在落盘…";
    }
    const vs = msg.v3_summary || {};
    appendSessionScoreSummary(el.summaryMsg, vs);
    if (el.verifyBadge) {
      el.verifyBadge.textContent = "落盘中…";
      el.verifyBadge.className = "na";
    }
  } else if (t === "session_saved") {
    starting = false;
    lastCampaignManifest = msg.campaign?.manifest || null;
    if (msg.campaign?.manifest) activeCampaign = msg.campaign.manifest;
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
    const ss = msg.v2_summary?.session_score ?? msg.v3_summary?.session_score;
    const ssMax = msg.v2_summary?.session_score_max ?? msg.v3_summary?.session_score_max;
    if (msg.v3_summary) {
      appendSessionScoreSummary(el.summaryMsg, msg.v3_summary);
    } else if (ss != null && ssMax != null) {
      el.summaryMsg.textContent += ` · 本场得分 ${formatScoreHalf(ss)}/${formatScoreHalf(ssMax)}`;
    }
    if (activeSubject || msg.subject_id) {
      if (!activeSubject && msg.subject_id) activeSubject = msg.subject_id;
      showFtPanel(msg);
    }
    if (msg.campaign) {
      activeCampaign = msg.campaign.manifest || activeCampaign;
      simCampaigns = simCampaigns.map((c) =>
        c.campaign_id === activeCampaign?.campaign_id ? activeCampaign : c,
      );
      fillSimCampaignSelect(simCampaigns, activeCampaign?.campaign_id || "");
      updateSimCampaignStatus();
      const campLine = msg.campaign.summary_path
        ? ` · Campaign 汇总 ${msg.campaign.summary_path}`
        : "";
      const nxt = msg.campaign.next_run;
      if (nxt) {
        applySimRunToForm(nxt);
        el.summaryMsg.textContent += ` · 下一场 ${nxt}`;
      } else if (msg.campaign.completed) {
        el.summaryMsg.textContent += " · Campaign 已完成";
      }
      if (campLine) el.summaryMsg.textContent += campLine;
    }
    if (activeSimMode) requestSimCampaignList();
    if (msg.model_presets) refreshModelPresetsFromServer(msg);
    if (msg.suggest_session_id && !activeSimMode) {
      activeSubjectInfo = {
        ...activeSubjectInfo,
        suggest_session_id: msg.suggest_session_id,
      };
      updateSubjectBar();
    }
    if (msg.sim_index) {
      activeSubjectInfo = {
        ...activeSubjectInfo,
        index: msg.sim_index,
        suggest_session_id: msg.suggest_session_id || msg.sim_index?.suggest_session_id,
        current_weights: msg.sim_index?.current_model || activeSubjectInfo?.current_weights,
      };
      updateSubjectBar();
      if (activeSimMode) {
        const nxtRun =
          msg.campaign?.next_run ||
          msg.suggest_session_id ||
          msg.sim_index?.suggest_session_id;
        if (nxtRun) applySimRunToForm(nxtRun);
        else refreshSimRunSuggestion();
      }
    }
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
      } else if (msg.phase_mode === "v3_session" || msg.phase_mode === "sim_v3_session") {
        const vs = msg.v3_summary || {};
        badge.textContent = `v3 ${vs.quality_tier || "—"}`;
        badge.className = vs.frozen ? "pass" : "warn";
        if (el.v3SummaryDetail) {
          el.v3SummaryDetail.classList.remove("hidden");
          const r = vs.report || {};
          const ov = r.overall || {};
          const wa = vs.window_acc ?? ov.acc_window;
          const waTxt = formatWindowAccPct(wa);
          const trialTxt = formatWindowAccPct(ov.acc_argmax);
          const parts = [
            `块顺序 ${(vs.block_order || []).join("→")}`,
            `frozen=${vs.frozen}`,
            r.quality_tier || vs.quality_tier || "",
          ];
          if (waTxt) {
            const n = vs.window_acc_n ?? ov.n_windows;
            parts.push(n != null ? `窗级识别率 ${waTxt}（${n} 窗）` : `窗级识别率 ${waTxt}`);
          }
          if (trialTxt) parts.push(`试次多数票 ${trialTxt}`);
          el.v3SummaryDetail.textContent = parts.filter(Boolean).join(" · ");
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
  } else if (t === "session_exclude_record_ack") {
    if (!msg.ok) {
      alert(msg.message || "标记失败");
      return;
    }
    sessionRecordExcluded = true;
    updateSessionNoRecordButton();
    if (msg.campaign) {
      activeCampaign = msg.campaign;
      simCampaigns = simCampaigns.map((c) =>
        c.campaign_id === activeCampaign?.campaign_id ? activeCampaign : c,
      );
      fillSimCampaignSelect(simCampaigns, activeCampaign?.campaign_id || "");
      updateSimCampaignStatus();
    }
    if (msg.next_run) applySimRunToForm(msg.next_run);
    if (msg.sim_index) {
      activeSubjectInfo = {
        ...activeSubjectInfo,
        index: msg.sim_index,
        suggest_session_id: msg.sim_index?.suggest_session_id,
      };
      updateSubjectBar();
    }
    if (msg.model_presets) refreshModelPresetsFromServer(msg);
    const run = msg.run_id ? ` · run ${msg.run_id} 已释放可重采` : "";
    if (el.summaryMsg) {
      el.summaryMsg.textContent += " · 本场不记入实验记录（文件保留，仍可用于微调）" + run;
    }
    if (el.ftStatus) {
      el.ftStatus.textContent = "本场已标记不记入 Campaign/索引；勾选上方 session 仍可进行微调";
    }
    if (activeSubject) send({ type: "subject_info", subject_id: activeSubject });
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
  const sim = Boolean(activeSimMode);
  const cfgOpen =
    sim || el.form.querySelector('[name="open_subject_page"]')?.checked !== false;
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
  if (!activeSimMode) {
    const sid = readSetupSessionId();
    if (el.setupSessionId) el.setupSessionId.value = sid;
    const bl = boardLabel(currentProtocolBoard());
    const boardHits = sessionsMatchingId(sid, { boardOnly: true });
    if (boardHits.length && !el.sessionOverwrite?.checked) {
      const sug = syncSuggestSessionIdForBoard() || "";
      if (!sug || sug.toLowerCase() === sid.toLowerCase()) {
        showErrors([
          `会话编号「${sid}」已存在，且无法算出可用新建号。请手改编号或勾选覆盖后重试。`,
        ]);
        return;
      }
      const useSug = confirm(
        `当前 ${bl}：会话编号「${sid}」已存在 ${boardHits.length} 场。\n\n` +
          `确定：改用新建 ${sug} 并开始实验\n取消：留在设置页（可选历史编号并确认覆盖）`,
      );
      if (!useSug) return;
      fillSessionIdSelect(sug);
      setSessionIdSelectValue(sug, { overwrite: false });
      // 落定新编号后继续开场（勿 return）
    }
    if (boardHits.length && el.sessionOverwrite?.checked) {
      const ok = confirm(
        `将覆盖 ${bl} 会话编号「${sid}」：旧目录移入 _archived，然后重新采集。\n确认继续？`,
      );
      if (!ok) return;
    }
  }
  const cfg = formToRunConfig();
  hotkeysEnabled = cfg.ui.operator_hotkeys !== false;
  showErrors([]);
  if (cfg.ui.remember_last_config) saveLocalDefaults(cfg);
  send({ type: "session_start", run_config: cfg });
}

el.form.addEventListener("change", () => {
  syncAcqUi();
  syncWorkModeUi();
  syncFtReplayRatioUi();
  if (isV2Mode() || isV3Mode()) {
    replayAdviceManualOverride.setup = false;
    updateReplayAdvice("setup", { autoApply: false });
    if (isV2Mode()) renderSetupTimelineV2();
    else renderSetupTimelineV3();
  } else {
    renderSetupTimeline();
  }
});
el.form.querySelector('[name="ft_use_replay"]')?.addEventListener("change", (ev) => {
  replayAdviceManualOverride.setup = true;
  syncFtReplayRatioUi();
  const ftRep = readFtReplayOptions(el.form);
  if (el.ftUseReplay) el.ftUseReplay.checked = ftRep.use_replay;
  if (el.ftReplayRatio) el.ftReplayRatio.value = String(ftRep.replay_ratio || DEFAULT_FT_REPLAY_RATIO);
});
el.ftUseReplay?.addEventListener("change", () => {
  replayAdviceManualOverride.panel = true;
  syncFtReplayRatioUi();
});
el.btnSetupReplayAdopt?.addEventListener("click", () => {
  replayAdviceManualOverride.setup = false;
  applyReplayAdvice(lastReplayAdvice.setup, "setup", { force: true });
});
el.btnFtReplayAdopt?.addEventListener("click", () => {
  replayAdviceManualOverride.panel = false;
  applyReplayAdvice(lastReplayAdvice.panel, "panel", { force: true });
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
document.getElementById("btn-v2-enter-game")?.addEventListener("click", () => {
  if (
    !confirm(
      "确认跳过引导 / 剩余标定 / 准入，直接进入游戏轮？\n（当前试次若正在进行会被打断）"
    )
  ) {
    return;
  }
  send({ type: "operator", action: "enter_game" });
  stopGuidanceCountdown();
  if (el.btnV2Guidance) el.btnV2Guidance.disabled = true;
  if (el.v2StageHint) el.v2StageHint.textContent = "已请求直接进入游戏…";
  showRunAlert("已请求直接进入游戏（跳过引导/标定/准入）", "gate-pass");
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
  requestAbortSession({ fromSetup: false });
});
document.getElementById("btn-setup-abort")?.addEventListener("click", () => {
  requestAbortSession({ fromSetup: true });
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
  resetFtPanel();
  showView("setup");
  maybeShowReuseBar(defaultsFromServer || loadLocalDefaults());
});

el.loginForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = new FormData(el.loginForm);
  const workMode = String(fd.get("login_work_mode") || "experiment");
  const simMode = workMode === "sim";
  const sid = simMode
    ? String(fd.get("login_sim_subject") || "A01").trim().toUpperCase()
    : String(fd.get("login_subject_id") || "").trim().toLowerCase();
  const display = String(fd.get("login_display_name") || "").trim();
  if (!sid) return;
  send({ type: "subject_login", subject_id: sid, display_name: display, sim_mode: simMode });
});

document.querySelectorAll('input[name="login_work_mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    const sim = document.querySelector('input[name="login_work_mode"]:checked')?.value === "sim";
    document.querySelectorAll(".login-exp").forEach((n) => n.classList.toggle("hidden", sim));
    document.querySelectorAll(".login-sim").forEach((n) => n.classList.toggle("hidden", !sim));
    const inp = document.getElementById("login-subject-id");
    if (inp) inp.required = !sim;
  });
});

document.getElementById("btn-subject-logout")?.addEventListener("click", () => {
  send({ type: "subject_logout" });
});

el.btnSimCampaignCreate?.addEventListener("click", () => {
  if (!activeSimMode || !activeSubject) {
    alert("请先登录仿真被试");
    return;
  }
  const checked = [];
  el.simRunQueue?.querySelectorAll(".sim-run-cb:checked").forEach((cb) => {
    if (cb.value) checked.push(cb.value);
  });
  if (!checked.length) {
    alert("请勾选至少一个 run");
    return;
  }
  const fd = new FormData(el.form);
  const replayAlign =
    String(
      (el.form.querySelector('input[name="sim_replay_align"]:checked') || {}).value || "schedule_align",
    ).trim() || "schedule_align";
  send({
    type: "sim_campaign_create",
    subject_id: activeSubject,
    session_queue: checked,
    session_trials_total: Number(fd.get("sim_trials_total") || 36),
    replay_align: replayAlign,
    replay_speed: Number(fd.get("sim_replay_speed") || 4),
    leave_next_mode: Boolean(el.form?.querySelector('[name="sim_leave_next_mode"]')?.checked),
  });
});

el.simCampaignSelect?.addEventListener("change", onSimCampaignSelectChange);
el.form?.querySelector('[name="sim_use_campaign_queue"]')?.addEventListener("change", updateSimCampaignStatus);
el.form?.querySelector('[name="sim_run_id"]')?.addEventListener("input", () => {
  if (!activeSimMode && !isSimV3Mode()) return;
  const v = String(el.form?.querySelector('[name="sim_run_id"]')?.value || "").trim().toLowerCase();
  if (v.startsWith("run") && el.setupSessionId) el.setupSessionId.value = v;
});

function bindFtAdvancedUi() {
  const onCh = () => syncFtAdvancedUi();
  el.ftEarlyStop?.addEventListener("change", onCh);
  el.form?.querySelector('[name="ft_early_stop"]')?.addEventListener("change", onCh);
}
bindFtAdvancedUi();
syncFtAdvancedUi();

el.ftLeaveNext?.addEventListener("change", () => {
  applyLeaveNextFtSelection();
  updateFtRampHint();
  updateReplayAdvice("panel");
});

el.setupSessionId?.addEventListener("change", () => onSessionIdSelectChange());
el.setupSessionId?.addEventListener("input", () => updateSessionIdConflictUi());
el.setupSessionId?.addEventListener("blur", () => {
  if (activeSimMode) return;
  const sid = readSetupSessionId();
  if (el.setupSessionId && sid) el.setupSessionId.value = sid;
  updateSessionIdConflictUi();
});
el.sessionIdSuggestBtn?.addEventListener("click", () => applySuggestedSessionId({ force: true }));
el.sessionOverwrite?.addEventListener("change", () => updateSessionIdConflictUi());

el.btnFtStart?.addEventListener("click", () => {
  const mode = document.querySelector('input[name="ft_mode"]:checked')?.value;
  if (mode !== "run") {
    if (el.ftStatus) el.ftStatus.textContent = "已跳过微调";
    return;
  }
  const leaveNext = Boolean(el.ftLeaveNext?.checked) && Boolean(activeSimMode);
  const paths = collectFtSessionPaths();
  if (!leaveNext && !paths.length) {
    if (el.ftLeaveNext?.checked && !activeSimMode) {
      alert(
        "Leave-Next 开启：本场不入 FT，且暂无合格历史 session。\n" +
          "可取消 Leave-Next 后勾选本场，或跳过微调。",
      );
      return;
    }
    alert("请至少勾选一个 session");
    return;
  }
  if (leaveNext && !activeCampaign?.manifest_path) {
    alert("Leave-Next 需要先选择 Campaign");
    return;
  }
  const evalRun = resolveFtEvalRunId();
  if (leaveNext && !evalRun) {
    alert("Leave-Next 无法确定 eval run（请确认刚结束的 session 或 run_id）");
    return;
  }
  if (leaveNext && activeCampaign?.session_queue) {
    const q = (activeCampaign.session_queue || []).map((r) => String(r).toLowerCase());
    const rStage = q.indexOf(evalRun);
    if (rStage === 0) {
      alert("R0 仅底座评估，不 FT（Leave-Next：eval 之前无 train session）");
      return;
    }
    if (rStage < 0) {
      alert(`eval run ${evalRun} 不在 Campaign 队列`);
      return;
    }
  }
  const ftRep = readFtReplayOptions(null);
  const ftAdv = readFtAdvancedOptions();
  send({
    type: "finetune_start",
    subject_id: activeSubject,
    session_paths: paths,
    exclude_invalid: Boolean(el.ftExcludeInvalid?.checked),
    use_replay: ftRep.use_replay,
    no_replay: !ftRep.use_replay,
    replay_ratio: ftRep.replay_ratio,
    early_stop: ftAdv.early_stop,
    max_epochs: ftAdv.max_epochs,
    patience: ftAdv.patience,
    epochs: ftAdv.fixed_epochs,
    deterministic: ftAdv.deterministic,
    seed: ftAdv.seed,
    leave_next_mode: leaveNext,
    eval_run_id: leaveNext ? evalRun : null,
    campaign_manifest: leaveNext ? activeCampaign?.manifest_path || null : null,
    use_ramp_replay_defaults: leaveNext,
  });
});

el.btnFtPromote?.addEventListener("click", () => {
  if (!lastFtRunDir) return;
  if (lastFtGatePass === false) {
    const ok = confirm(
      "本轮门控 FAIL（常见原因：训练准确率远高于 heldout，疑似过拟合）。\n\n" +
        "仍要用这组权重替换 current 吗？",
    );
    if (!ok) return;
  }
  send({
    type: "finetune_promote",
    subject_id: activeSubject,
    ft_run_dir: lastFtRunDir,
    reason: lastFtGatePass === false ? "operator_force_despite_gate_fail" : "operator_summary_confirm",
  });
});

el.btnFtKeep?.addEventListener("click", () => {
  if (el.ftStatus) el.ftStatus.textContent = "已保留旧 current；FT 快照仍在 ft_runs";
  if (el.btnFtPromote) el.btnFtPromote.classList.add("hidden");
  if (el.btnFtKeep) el.btnFtKeep.classList.add("hidden");
});

el.btnNextSession?.addEventListener("click", () => {
  if (ftBusy) {
    alert("微调进行中，请稍候");
    return;
  }
  starting = false;
  resetRunView();
  lockedConfig = null;
  sessionRoot = "";
  sessionRecordExcluded = false;
  lastCampaignManifest = null;
  resetFtPanel();
  if (activeSimMode) {
    refreshSimRunSuggestion();
  } else if (activeSubjectInfo?.suggest_session_id) {
    fillSessionIdSelect(activeSubjectInfo.suggest_session_id);
    setSessionIdSelectValue(activeSubjectInfo.suggest_session_id, { overwrite: false });
  }
  showView("setup");
});

el.btnSessionNoRecord?.addEventListener("click", () => {
  if (!sessionRoot) return;
  if (sessionRecordExcluded) return;
  const ok = confirm(
    "不记入实验记录？\n\n· 本场文件仍保留，可勾选用于微调\n· 从 Campaign 汇总与被试索引中移除\n· 若属 Campaign 队列，对应 run 可重新采集",
  );
  if (!ok) return;
  send({
    type: "session_exclude_record",
    session_root: sessionRoot,
    subject_id: activeSubject,
    campaign_manifest: lastCampaignManifest || activeCampaign || null,
  });
  if (el.ftStatus) el.ftStatus.textContent = "正在标记不记入记录…";
});

window.addEventListener("keydown", (e) => {
  if (!hotkeysEnabled) return;
  if (e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
  // 设置页也允许 Esc 中止卡住的后台会话
  if (e.key === "Escape") {
    const onRun = !el.run.classList.contains("hidden");
    const onSetup = !el.setup.classList.contains("hidden");
    if (onRun || onSetup) {
      requestAbortSession({ fromSetup: onSetup && !onRun });
    }
    return;
  }
  if (el.run.classList.contains("hidden")) return;
  const k = e.key.toLowerCase();
  if (k === "b") send({ type: "operator", action: "split_session" });
  if (k === "q") send({ type: "questionnaire_open" });
  if (k === "p") send({ type: "operator", action: "toggle_pause" });
  if (k === "n") send({ type: "operator", action: "continue" });
  if (k === "g") send({ type: "operator", action: "gate_ok" });
  if (k === "r") send({ type: "operator", action: "reject" });
});

for (const prefix of ["v2", "v3"]) {
  document.getElementById(`${prefix}-votes-prev`)?.addEventListener("click", () => {
    navigateVoteHistory(prefix, -1);
  });
  document.getElementById(`${prefix}-votes-next`)?.addEventListener("click", () => {
    navigateVoteHistory(prefix, 1);
  });
  bindVoteHistorySwipe(prefix);
}

const hash = (location.hash || "").replace("#", "");
const views = ["login", "setup", "run", "summary"];
if (hash && views.includes(hash) && hash !== "login") {
  showView(hash);
} else if (!loadSubjectLoginLocal()) {
  showView("login");
}
syncAcqUi();
syncWorkModeUi();
syncFtReplayRatioUi();
updateReplayAdvice("setup", { autoApply: false });
renderSetupTimeline();
connect();
