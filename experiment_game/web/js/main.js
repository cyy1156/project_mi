import { WsClient } from "./ws_client.js?v=20260825arm1";
import { HomeDeskScene } from "./scene.js?v=20260825arm1";
import { handleV2Stage, maybeDemo, setSubjectFeedbackMode } from "./v2_bridge.js?v=20260827m";

const params = new URLSearchParams(location.search);
const wsUrl = params.get("ws") || `ws://${location.hostname || "127.0.0.1"}:8765`;

const el = {
  cross: document.getElementById("cross"),
  text: document.getElementById("hud-text"),
  sub: document.getElementById("hud-sub"),
  phase: document.getElementById("phase-tag"),
  status: document.getElementById("status"),
  helpTip: document.getElementById("help-tip"),
  prompt: document.getElementById("prompt"),
  promptTitle: document.getElementById("prompt-title"),
  promptBody: document.getElementById("prompt-body"),
  promptBtn: document.getElementById("prompt-btn"),
  promptHint: document.getElementById("prompt-hint"),
  qWrap: document.getElementById("questionnaire"),
  qTitle: document.getElementById("q-title"),
  qBody: document.getElementById("q-body"),
  qSubmit: document.getElementById("q-submit"),
  qClose: document.getElementById("q-close"),
  qCloseBottom: document.getElementById("q-close-bottom"),
  qHint: document.getElementById("q-hint"),
  offline: document.getElementById("offline"),
  opbar: document.getElementById("opbar"),
  opState: document.getElementById("op-state"),
  opPause: document.getElementById("op-pause"),
  opContinue: document.getElementById("op-continue"),
  opReject: document.getElementById("op-reject"),
  opAbort: document.getElementById("op-abort"),
};

const scene = new HomeDeskScene(document.getElementById("c"));
window.__miScene = scene;
window.__v2scene = {
  fixation: () => scene.v2Fixation(),
  cue: (label) => scene.v2Cue(label),
  gameLevel: (n, reach) => scene.v2GameLevel(n, reach),
  iti: () => scene.v2Iti(),
  idle: (text) => scene.v2Idle(text),
};

/** @type {"idle"|"phase2"|"v2_session"|"v3_session"|"v4_session"} */
let sessionMode = "idle";
let promptOpen = false;
let promptAllowSubject = true;
let sessionDone = false;
let qOpen = false;
let paused = false;
let lastPromptPayload = null;
let promptContinuePending = false;
let promptContinueSentAt = 0;
let dismissedPromptId = null;
/** @type {import("./ws_client.js").WsClient | null} */
let client = null;

function setHelpTip(text) {
  if (el.helpTip) el.helpTip.textContent = text;
}

function setStatus(s) {
  if (el.status) el.status.textContent = s;
  const offline =
    /断开|错误|重试|服务已结束|连接 WebSocket/i.test(s) && !sessionDone;
  setOffline(offline && !promptOpen && !qOpen);
  if (qOpen) {
    setHelpTip("请完成后点击「提交问卷」");
  } else if (sessionDone) {
    setHelpTip("本会话已结束；操作台点「问卷」可在此页作答");
  } else if (promptOpen) {
    setHelpTip(
      promptAllowSubject
        ? "请点击「继续」或按空格 / Enter"
        : "请操作者确认（G / 代确认）"
    );
  } else if (paused) {
    setHelpTip("已暂停 — 操作者按 P 恢复");
  } else if (offline) {
    setHelpTip("请先运行 open_operator.bat，再刷新本页");
  } else {
    setHelpTip("被试：空格确认 · 操作者：P/N/G/R/Esc");
  }
}

function setOffline(on) {
  if (!el.offline) return;
  el.offline.classList.toggle("hidden", !on);
  el.offline.setAttribute("aria-hidden", on ? "false" : "true");
}

function clearIdleOverlay() {
  try {
    scene._v2IdleEl?.remove();
    scene._v2IdleEl = null;
  } catch {
    /* ignore */
  }
  document.getElementById("v2-idle-overlay")?.remove();
  document.getElementById("v2ov")?.remove();
}

function resetSubjectScene(modeLabel) {
  clearIdleOverlay();
  if (el.text) el.text.textContent = modeLabel || "";
  if (el.sub) el.sub.textContent = "";
  if (el.cross) el.cross.classList.add("hidden");
}

function normalizePhaseMode(pm) {
  const m = pm || "phase2_full";
  if (m === "sim_v3_session") return "v3_session";
  return m;
}

function applySessionMode(mode, label) {
  const next = mode || "idle";
  if (next !== sessionMode) {
    sessionMode = next;
    const titles = {
      phase2: "Phase2 诱导",
      v2_session: "v2 会话",
      v3_session: "v3 探针",
      v4_session: "v4 质量检测",
      idle: "",
    };
    // 只改 HUD，不盖全屏遮罩（遮罩会挡住后续「静息基线」等真流程）
    clearIdleOverlay();
    if (el.text) el.text.textContent = label || titles[next] || "";
    if (el.sub) el.sub.textContent = next === "idle" ? "" : "连接中…";
    if (el.cross) el.cross.classList.add("hidden");
    if (el.phase) el.phase.textContent = next === "idle" ? "" : next;
  } else if (label && el.phase) {
    el.phase.textContent = label;
  }
}

function isV2Family() {
  return sessionMode === "v2_session" || sessionMode === "v3_session";
}

function showPrompt(msg) {
  if (!el.prompt) return;
  const pid = msg?.id || null;
  if (pid && pid === dismissedPromptId) return;
  if (
    promptContinuePending &&
    Date.now() - promptContinueSentAt < 4000
  ) {
    return;
  }
  lastPromptPayload = msg;
  promptContinuePending = false;
  dismissedPromptId = null;
  sessionDone = false; // 新弹窗出现时允许确认（避免上场 done 卡住）
  if (el.promptTitle) el.promptTitle.textContent = msg.title || "";
  if (el.promptBody) el.promptBody.textContent = msg.body || "";
  if (el.promptBtn) {
    el.promptBtn.textContent = msg.button || "继续";
    el.promptBtn.disabled = false;
  }
  promptAllowSubject = msg.allow_subject !== false;
  if (el.promptHint) {
    el.promptHint.innerHTML = promptAllowSubject
      ? "也可按 <kbd>空格</kbd> 或 <kbd>Enter</kbd>"
      : "被试空格无效 · 操作者按 <kbd>G</kbd> / <kbd>N</kbd>，或点本按钮";
  }
  el.prompt.classList.remove("hidden");
  el.prompt.setAttribute("aria-hidden", "false");
  promptOpen = true;
  setOffline(false);
  setStatus((el.status && el.status.textContent) || "已连接");
  try {
    el.promptBtn && el.promptBtn.focus();
  } catch {
    /* ignore */
  }
}

function hidePrompt() {
  if (!el.prompt) return;
  el.prompt.classList.add("hidden");
  el.prompt.setAttribute("aria-hidden", "true");
  promptOpen = false;
  if (el.promptBtn) el.promptBtn.disabled = false;
}

function sendContinue(role = "subject") {
  if (!promptOpen && !lastPromptPayload) return;
  if (role === "subject" && !promptAllowSubject) return;
  if (!client) {
    setStatus("未连接 — 无法确认");
    setHelpTip("请保持操作台黑窗口打开；刷新本页或重新打开诱导页");
    setOffline(true);
    return;
  }
  if (el.promptBtn) el.promptBtn.disabled = true;
  dismissedPromptId = lastPromptPayload?.id || dismissedPromptId;
  promptContinuePending = true;
  promptContinueSentAt = Date.now();
  hidePrompt();
  const ok = client.send({ type: "continue", role });
  if (!ok) {
    promptContinuePending = false;
    if (lastPromptPayload) showPrompt(lastPromptPayload);
    if (el.promptBtn) el.promptBtn.disabled = false;
    setStatus("未连接 — 无法确认");
    setHelpTip("请保持操作台黑窗口打开；刷新本页或重新打开诱导页");
    setOffline(true);
    return;
  }
  if (!promptAllowSubject) {
    client.send({ type: "operator", action: "gate_ok" });
  }
  setStatus("已确认，请稍候…");
}

function sendOperator(action) {
  if (!client) return;
  client.send({ type: "operator", action });
  // 确认类操作立刻关弹窗，避免后端已前进、前端仍挡着
  if (action === "gate_ok" || action === "continue") {
    hidePrompt();
  }
}

function updateOpState(msg) {
  paused = !!msg.paused;
  if (el.opbar) el.opbar.classList.toggle("paused", paused);
  if (el.opPause) el.opPause.textContent = paused ? "恢复" : "暂停";
  if (el.opState) {
    const parts = [
      `phase=${msg.phase || "—"}`,
      `trial=${msg.trial_id ?? "—"}`,
      `label=${msg.label ?? "—"}`,
      `obj=${msg.object || "—"}`,
      `scene=${msg.scene || "—"}`,
      `reject=${msg.reject_count ?? 0}`,
    ];
    if (paused) parts.unshift("PAUSED");
    el.opState.textContent = parts.join(" · ");
  }
  setStatus((el.status && el.status.textContent) || "已连接");
}

/* ---------------- 采集结束问卷（操作者 Q 推送） ---------------- */

function showQuestionnaire(msg) {
  // 问卷 deliberately 在会话结束后由操作台推送；不可因 sessionDone 拦截
  if (!el.qWrap) return;
  el.qTitle.textContent = msg.title || "问卷";
  el.qBody.innerHTML = "";
  el.qHint.textContent = msg.session_root
    ? `提交后保存到：${msg.session_root}/99_summary/`
    : "";
  let lastGroup = "";
  for (const q of msg.questions || []) {
    if (q.group && q.group !== lastGroup) {
      lastGroup = q.group;
      const g = document.createElement("div");
      g.className = "q-group";
      g.textContent = lastGroup;
      el.qBody.appendChild(g);
    }
    const item = document.createElement("div");
    item.className = "q-item";
    const p = document.createElement("p");
    p.textContent = q.text || q.id;
    item.appendChild(p);
    const opts = document.createElement("div");
    opts.className = "q-options";
    if (q.kind === "scale5") {
      for (let v = 1; v <= 5; v++) {
        const lab = document.createElement("label");
        const rb = document.createElement("input");
        rb.type = "radio";
        rb.name = `q_${q.id}`;
        rb.value = String(v);
        rb.dataset.qid = q.id;
        lab.appendChild(rb);
        lab.appendChild(document.createTextNode(`${v}分`));
        opts.appendChild(lab);
      }
      const anchors = document.createElement("div");
      anchors.className = "q-anchors";
      const [lo, hi] = q.anchors || ["1 = 低", "5 = 高"];
      anchors.textContent = `${lo} · ${hi}`;
      item.appendChild(opts);
      item.appendChild(anchors);
    } else if (q.kind === "choice") {
      for (const opt of q.options || []) {
        const lab = document.createElement("label");
        const rb = document.createElement("input");
        rb.type = "radio";
        rb.name = `q_${q.id}`;
        rb.value = opt;
        rb.dataset.qid = q.id;
        lab.appendChild(rb);
        lab.appendChild(document.createTextNode(opt));
        opts.appendChild(lab);
      }
      item.appendChild(opts);
    } else {
      const ta = document.createElement("textarea");
      ta.rows = 2;
      ta.style.width = "100%";
      ta.dataset.qid = q.id;
      ta.dataset.kind = "text";
      item.appendChild(ta);
    }
    el.qBody.appendChild(item);
  }
  el.qWrap.classList.remove("hidden");
  el.qWrap.setAttribute("aria-hidden", "false");
  qOpen = true;
  if (el.qSubmit) el.qSubmit.disabled = false;
  setHelpTip("请完成后点击「提交问卷」");
  setStatus("问卷进行中");
}

function hideQuestionnaire() {
  if (!el.qWrap) return;
  el.qWrap.classList.add("hidden");
  el.qWrap.setAttribute("aria-hidden", "true");
  qOpen = false;
  if (el.qSubmit) el.qSubmit.disabled = false;
  if (sessionDone) {
    setStatus("完成");
    setHelpTip("本会话已结束；操作台点「问卷」可再次打开");
  }
}

function closeQuestionnaireManual() {
  hideQuestionnaire();
  if (el.qHint) el.qHint.textContent = "";
}

function submitQuestionnaire() {
  if (!qOpen) return;
  const answers = {};
  const missing = [];
  document.querySelectorAll("#q-body input[type=radio]:checked").forEach((rb) => {
    answers[rb.dataset.qid] = rb.value;
  });
  document.querySelectorAll("#q-body textarea").forEach((ta) => {
    if (ta.value.trim()) answers[ta.dataset.qid] = ta.value.trim();
  });
  // 漏填检查：每个单选组必须有选中
  const groups = new Set(
    [...document.querySelectorAll("#q-body input[type=radio]")].map(
      (rb) => rb.name
    )
  );
  for (const name of groups) {
    if (!document.querySelector(`#q-body input[name="${CSS.escape(name)}"]:checked`)) {
      missing.push(name.replace("q_", ""));
    }
  }
  if (missing.length) {
    if (el.qHint) el.qHint.textContent = `还有未作答的题目：${missing.join("、")}`;
    return;
  }
  if (el.qSubmit) el.qSubmit.disabled = true;
  if (el.qHint) el.qHint.textContent = "提交中…";
  if (!client) {
    if (el.qSubmit) el.qSubmit.disabled = false;
    if (el.qHint) el.qHint.textContent = "未连接，无法提交";
    return;
  }
  client.send({ type: "questionnaire_result", form: "post", answers });
}

el.qClose?.addEventListener("click", closeQuestionnaireManual);
el.qCloseBottom?.addEventListener("click", closeQuestionnaireManual);
if (el.qSubmit) {
  el.qSubmit.addEventListener("click", submitQuestionnaire);
}

if (maybeDemo()) {
  // v2 演示模式：渲染层兜底，不启动 ws
} else {
client = new WsClient(
  wsUrl,
  (msg) => {
    if (msg.type === "session_started") {
      const pm = normalizePhaseMode(msg.phase_mode || "phase2_full");
      const mapped =
        pm === "v2_session" || pm === "v3_session" || pm === "v4_session"
          ? pm
          : "phase2";
      applySessionMode(mapped, pm === "v3_session" && msg.phase_mode === "sim_v3_session" ? "仿真 v3" : undefined);
      setSubjectFeedbackMode(msg.subject_feedback_mode || "none");
      sessionDone = false;
      setOffline(false);
      setStatus("会话已启动");
      return;
    }
    if (msg.type === "v2_stage") {
      // v1 Phase2 的 stage 动画不得混入；仅 v2/v3 消费
      if (sessionMode === "phase2") return;
      if (sessionMode === "idle") applySessionMode("v2_session");
      if (sessionMode === "v4_session") return;
      clearIdleOverlay();
      handleV2Stage(msg.stage, msg.ctx, msg.data);
      return;
    }
    if (msg.type === "v4_start") {
      applySessionMode("v4_session", "v4 质量检测");
      clearIdleOverlay();
      if (el.text) el.text.textContent = "数据质量检测";
      if (el.sub) {
        el.sub.textContent = "请静坐，等待操作台格子变绿";
      }
      if (el.cross) el.cross.classList.remove("hidden");
      return;
    }
    if (msg.type === "v4_pass") {
      clearIdleOverlay();
      if (el.text) el.text.textContent = "质量达标";
      if (el.sub) el.sub.textContent = "可开始 v3 / v2 正式实验";
      return;
    }
    if (msg.type === "v4_summary") {
      clearIdleOverlay();
      if (el.text) el.text.textContent = "质量检测结束";
      if (el.sub) {
        el.sub.textContent = `结论：${msg.verdict || msg.rolling_verdict || "—"}`;
      }
      return;
    }
    if (msg.type === "hud") {
      clearIdleOverlay();
      if (el.text) el.text.textContent = msg.text || "";
      if (el.sub) el.sub.textContent = msg.subtext || "";
      if (el.cross) el.cross.classList.toggle("hidden", !msg.show_cross);
      if (msg.text) setStatus("进行中");
    } else if (msg.type === "stage") {
      // v2/v3/v4 会话忽略 v1 Phase2 舞台消息，避免画面串台
      if (isV2Family() || sessionMode === "v4_session") return;
      if (sessionMode === "idle") applySessionMode("phase2");
      if (promptOpen && msg.stage && msg.stage !== "idle") {
        hidePrompt();
      }
      if (msg.phase && el.phase) {
        const step = msg.learn_step ? ` · step ${msg.learn_step}` : "";
        el.phase.textContent = `${msg.phase}${step}`;
      }
      scene.applyStage(msg);
    } else if (msg.type === "questionnaire") {
      showQuestionnaire(msg);
    } else if (msg.type === "questionnaire_ack") {
      if (msg.ok) {
        if (el.qHint) {
          el.qHint.textContent = msg.path
            ? `已保存：${msg.path}`
            : "已提交，感谢配合";
        }
        setHelpTip(msg.path ? `问卷已保存：${msg.path}` : "问卷已提交");
        hideQuestionnaire();
        if (sessionDone) setStatus("完成");
      } else {
        if (el.qSubmit) el.qSubmit.disabled = false;
        const errText =
          (msg.errors && msg.errors.length)
            ? msg.errors.join("；")
            : (msg.message || "未知错误");
        if (el.qHint) el.qHint.textContent = `提交未通过：${errText}`;
      }
    } else if (msg.type === "prompt") {
      showPrompt(msg);
    } else if (msg.type === "continue_ack") {
      promptContinuePending = false;
      hidePrompt();
      setStatus("已连接");
    } else if (msg.type === "operator_state") {
      updateOpState(msg);
    } else if (msg.type === "session") {
      if (msg.status === "done" || msg.status === "aborted") {
        sessionDone = true;
        hidePrompt();
        if (el.text) el.text.textContent = "本会话结束";
        if (el.sub) el.sub.textContent = "可以关闭页面";
        setOffline(false);
        setStatus("完成");
        applySessionMode("idle");
      } else if (msg.status === "error") {
        setStatus(`错误: ${msg.message || ""}`);
      } else if (msg.status === "running" || msg.status === "gate") {
        sessionDone = false;
        const ph = normalizePhaseMode(msg.phase || "");
        if (ph === "v2_session" || ph === "v3_session" || ph === "v4_session") {
          applySessionMode(ph);
        } else if (ph && sessionMode === "idle") {
          applySessionMode("phase2", ph);
        } else if (ph && el.phase) {
          el.phase.textContent = msg.phase || ph;
        }
      } else if (msg.phase && el.phase) {
        el.phase.textContent = msg.phase;
      }
    } else if (msg.type === "hello") {
      sessionDone = false;
      setOffline(false);
      setStatus("已就绪，等待流程…");
      client.send({ type: "ready" });
    }
  },
  setStatus
);

if (el.promptBtn) {
  el.promptBtn.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    // 准入弹窗不允许被试：按钮按操作者确认
    sendContinue(promptAllowSubject ? "subject" : "operator");
  });
}
if (el.prompt) {
  el.prompt.style.pointerEvents = "auto";
  el.prompt.style.zIndex = "30";
}
if (el.opPause) {
  el.opPause.addEventListener("click", () => sendOperator("toggle_pause"));
}
if (el.opContinue) {
  el.opContinue.addEventListener("click", () => {
    if (promptOpen) sendContinue("operator");
    else sendOperator("continue");
  });
}
if (el.opReject) {
  el.opReject.addEventListener("click", () => sendOperator("reject"));
}
if (el.opAbort) {
  el.opAbort.addEventListener("click", () => {
    if (window.confirm("确认紧急结束本会话？")) sendOperator("abort");
  });
}

window.addEventListener("keydown", (ev) => {
  if (ev.repeat) return;
  const tag = (ev.target && ev.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;

  // 被试：空格/Enter 仅确认「允许被试」的 prompt
  if (ev.code === "Space" || ev.code === "Enter") {
    if (!promptOpen) return;
    ev.preventDefault();
    if (!promptAllowSubject) {
      // 准入：空格改为操作者确认，避免卡住
      sendContinue("operator");
      return;
    }
    sendContinue("subject");
    return;
  }

  // 操作者
  if (ev.code === "KeyP") {
    ev.preventDefault();
    sendOperator("toggle_pause");
  } else if (ev.code === "KeyN") {
    ev.preventDefault();
    if (promptOpen) sendContinue("operator");
    else sendOperator("continue");
  } else if (ev.code === "KeyG") {
    ev.preventDefault();
    sendOperator("gate_ok");
  } else if (ev.code === "KeyR") {
    ev.preventDefault();
    sendOperator("reject");
  } else if (ev.code === "Escape") {
    ev.preventDefault();
    if (window.confirm("确认紧急结束本会话？")) sendOperator("abort");
  }
});

window.addEventListener("focus", () => {
  if (promptOpen || promptContinuePending) return;
  client.send({ type: "sync" });
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    if (promptOpen || promptContinuePending) return;
    client.send({ type: "sync" });
  }
});

client.connect();

/* 渲染质量遥测：每 5s 上报 fps / 最大帧间隔，供后端写入 events.jsonl。
   画面掉帧会影响诱导稳定性，事后可按 trial 时间段追溯。 */
const STATS_INTERVAL_MS = 5000;
let statsFrames = 0;
let statsMaxGap = 0;
let statsLast = performance.now();

function reportStats() {
  const fps = (statsFrames * 1000) / STATS_INTERVAL_MS;
  client.send({
    type: "client_stats",
    fps: Math.round(fps * 10) / 10,
    max_gap_ms: Math.round(statsMaxGap),
  });
  statsFrames = 0;
  statsMaxGap = 0;
}

function loop(now) {
  requestAnimationFrame(loop);
  if (now - statsLast > 0) {
    statsMaxGap = Math.max(statsMaxGap, now - statsLast);
  }
  statsFrames += 1;
  statsLast = now;
  if (now >= (loop._nextReport || 0)) {
    loop._nextReport = now + STATS_INTERVAL_MS;
    if (statsFrames > 0) reportStats();
  }
  scene.update();
}
loop(performance.now());
}
