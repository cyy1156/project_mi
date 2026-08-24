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
};

/* ---------------- 试次时序：读取 / 时间轴渲染 ---------------- */

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

function syncPhaseModeUi() {
  const v2 =
    (el.form.querySelector('input[name="phase_mode"]:checked') || {}).value === "v2_session";
  document.querySelectorAll(".phase2-only").forEach((node) => {
    node.classList.toggle("hidden", v2);
  });
}

function setV2RunPanel(on) {
  el.v2Panel?.classList.toggle("hidden", !on);
  el.phaseSteps?.classList.toggle("hidden", on);
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
}

el.form.querySelectorAll('input[name="phase_mode"]').forEach((r) => {
  r.addEventListener("change", syncPhaseModeUi);
});

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
let hotkeysEnabled = true;
let lockedConfig = null;

function showView(name) {
  el.setup.classList.toggle("hidden", name !== "setup");
  el.run.classList.toggle("hidden", name !== "run");
  el.summary.classList.toggle("hidden", name !== "summary");
  location.hash = name;
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
      seed: seedRaw === "" ? null : Number(seedRaw),
      open_subject_page: el.form.querySelector('[name="open_subject_page"]').checked,
      skip_adapt: el.form.querySelector('[name="skip_adapt"]')?.checked || false,
      skip_learn: el.form.querySelector('[name="skip_learn"]')?.checked || false,
      skip_gate: el.form.querySelector('[name="skip_gate"]')?.checked || false,
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
  const pm = cfg.experiment?.phase_mode || "phase2_full";
  for (const r of el.form.querySelectorAll('input[name="phase_mode"]')) {
    r.checked = r.value === pm;
  }
  syncPhaseModeUi();
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
  el.runSummary.innerHTML = msg.phase_mode === "v2_session"
    ? [
        `<div><span class="k">模式</span>v2 会话</div>`,
        `<div><span class="k">采集</span>${msg.acq_enabled ? "开" : "关"}</div>`,
        `<div><span class="k">参数</span>config/v2_session.yaml</div>`,
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
      el.runTimingHint.textContent =
        `注视 ${msg.timing.fixation_s}s → 提示 ${msg.timing.cue_s}s → ` +
        `MI ${msg.timing.mi_s}s → 保持 ${msg.timing.post_mi_hold_s}s → ` +
        `静息 ${msg.timing.rest_s}s → 过渡 ${msg.timing.transition_s}s` +
        (msg.trial_total_s ? ` · 合计 ${msg.trial_total_s}s` : "");
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
      el.popupWarn.classList.remove("hidden");
      tryOpenSubject();
    }
  } else if (t === "session_started") {
    if (msg.subject_url) subjectUrl = msg.subject_url;
    sessionRoot = msg.session_root || "";
    const v2 = msg.phase_mode === "v2_session";
    setV2RunPanel(v2);
    if (v2 && el.v2StageHint) {
      el.v2StageHint.textContent = "v2 会话已启动；轮间请点「确认动觉引导完成」";
      setPhaseStep("adapt");
    }
    updateRunLockSummary(msg);
    // 诱导页已在 config_ack 打开；此处不再重复（避免多窗口）
  } else if (t === "v2_gate") {
    updateV2Gate(msg);
  } else if (t === "v2_stage") {
    if (el.v2StageHint) {
      const stage = msg.stage || "—";
      const mode = msg.ctx?.mode || "";
      el.v2StageHint.textContent = `v2 · ${stage}${mode ? ` (${mode})` : ""}`;
    }
    if (msg.stage === "guidance_begin" && el.btnV2Guidance) {
      el.btnV2Guidance.disabled = false;
    }
  } else if (t === "acq_status") {
    el.stAcq.textContent = `${msg.state || "—"}${msg.message ? " · " + msg.message : ""}`;
  } else if (t === "stage") {
    el.stPhase.textContent = msg.phase || "—";
    el.stStage.textContent = msg.stage || "—";
    el.stTrial.textContent = msg.trial_id ?? "—";
    el.stLabel.textContent = msg.label ?? "—";
    el.stObject.textContent = msg.object || "—";
    el.stScene.textContent = msg.scene || "—";
    if (msg.phase) setPhaseStep(msg.phase === "waiting_ready" ? "adapt" : msg.phase);
  } else if (t === "session") {
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
      if (msg.verify && msg.verify.passed === true) {
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
  const cfg = formToRunConfig();
  hotkeysEnabled = cfg.ui.operator_hotkeys !== false;
  showErrors([]);
  if (cfg.ui.remember_last_config) saveLocalDefaults(cfg);
  send({ type: "session_start", run_config: cfg });
}

el.form.addEventListener("change", () => {
  syncAcqUi();
  renderSetupTimeline();
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
});
document.getElementById("btn-summary-questionnaire")?.addEventListener("click", () => {
  send({ type: "questionnaire_open" });
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
  if (el.btnV2Guidance) el.btnV2Guidance.disabled = true;
  if (el.v2StageHint) el.v2StageHint.textContent = "已确认引导，会话继续…";
});
document.getElementById("btn-reject").addEventListener("click", () => {
  send({ type: "operator", action: "reject" });
});
document.getElementById("btn-reopen").addEventListener("click", () => {
  tryOpenSubject(true);
  send({ type: "open_subject_page" });
});
document.getElementById("btn-abort").addEventListener("click", () => {
  if (confirm("确认中止本场实验？已写入数据将尽量保留。")) {
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
