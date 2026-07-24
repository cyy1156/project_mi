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
};

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
      acquire_trials: Number(fd.get("acquire_trials") || 40),
      learn_trials_per_step: Number(fd.get("learn_trials_per_step") || 2),
      seed: seedRaw === "" ? null : Number(seedRaw),
      open_subject_page: el.form.querySelector('[name="open_subject_page"]').checked,
      skip_adapt: el.form.querySelector('[name="skip_adapt"]')?.checked || false,
      skip_learn: el.form.querySelector('[name="skip_learn"]')?.checked || false,
      skip_gate: el.form.querySelector('[name="skip_gate"]')?.checked || false,
      ready_timeout_s: 90,
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
  el.runSummary.innerHTML = [
    `<div><span class="k">采集</span>${msg.acq_enabled ? "开" : "关"}</div>`,
    `<div><span class="k">板卡</span>${msg.board_mode}${msg.board_mode === "cyton" ? " / " + (msg.serial_port || "") : ""}</div>`,
    `<div><span class="k">正式 trials</span>${msg.acquire_trials}</div>`,
    `<div><span class="k">会话目录</span><code>${msg.session_root || "—"}</code></div>`,
  ].join("");
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
    el.phase4Msg.textContent =
      `Phase4 OK · N=${s.n ?? "—"} · X=${JSON.stringify(s.X_shape || [])} · ` +
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
    updateRunLockSummary(msg);
    tryOpenSubject();
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
  } else if (t === "subject_page_opened") {
    if (!msg.ok) el.popupWarn.classList.remove("hidden");
  }
}

function tryOpenSubject() {
  const cfgOpen = el.form.querySelector('[name="open_subject_page"]')?.checked !== false;
  if (!cfgOpen) return;
  const w = window.open(subjectUrl, "_blank");
  if (!w) {
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
  const cfg = formToRunConfig();
  hotkeysEnabled = cfg.ui.operator_hotkeys !== false;
  showErrors([]);
  if (cfg.ui.remember_last_config) saveLocalDefaults(cfg);
  send({ type: "session_start", run_config: cfg });
}

el.form.addEventListener("change", syncAcqUi);
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

document.getElementById("btn-pause").addEventListener("click", () => {
  send({ type: "operator", action: paused ? "resume" : "pause" });
});
document.getElementById("btn-continue").addEventListener("click", () => {
  send({ type: "operator", action: "continue" });
});
document.getElementById("btn-gate").addEventListener("click", () => {
  send({ type: "operator", action: "gate_ok" });
});
document.getElementById("btn-reject").addEventListener("click", () => {
  send({ type: "operator", action: "reject" });
});
document.getElementById("btn-reopen").addEventListener("click", () => {
  tryOpenSubject();
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
  send({ type: "run_phase4", path: sessionRoot });
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
connect();
