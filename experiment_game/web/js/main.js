import { WsClient } from "./ws_client.js?v=20260821a";
import { HomeDeskScene } from "./scene.js?v=20260823b";
import { handleV2Stage, maybeDemo } from "./v2_bridge.js?v=20260823b";

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
  calProgress: (p) => scene.v2CalProgress(p),
  gameLevel: (n, reach) => scene.v2GameLevel(n, reach),
  iti: () => scene.v2Iti(),
  idle: (text) => scene.v2Idle(text),
};
let promptOpen = false;
let promptAllowSubject = true;
let sessionDone = false;
let paused = false;

function setHelpTip(text) {
  if (el.helpTip) el.helpTip.textContent = text;
}

function setStatus(s) {
  if (el.status) el.status.textContent = s;
  const offline =
    /断开|错误|重试|服务已结束|连接 WebSocket/i.test(s) && !sessionDone;
  setOffline(offline && !promptOpen);
  if (sessionDone) {
    setHelpTip("本会话已结束，可关闭页面");
  } else if (promptOpen) {
    setHelpTip(
      promptAllowSubject
        ? "请点击「继续」或按空格 / Enter"
        : "请操作者确认（G / 代确认）"
    );
  } else if (paused) {
    setHelpTip("已暂停 — 操作者按 P 恢复");
  } else if (offline) {
    setHelpTip("请先运行 open_induction.bat，再刷新本页");
  } else {
    setHelpTip("被试：空格确认 · 操作者：P/N/G/R/Esc");
  }
}

function setOffline(on) {
  if (!el.offline) return;
  el.offline.classList.toggle("hidden", !on);
  el.offline.setAttribute("aria-hidden", on ? "false" : "true");
}

function showPrompt(msg) {
  if (!el.prompt) return;
  if (el.promptTitle) el.promptTitle.textContent = msg.title || "";
  if (el.promptBody) el.promptBody.textContent = msg.body || "";
  if (el.promptBtn) el.promptBtn.textContent = msg.button || "继续";
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
}

function sendContinue(role = "subject") {
  if (!promptOpen || sessionDone) return;
  if (role === "subject" && !promptAllowSubject) return;
  hidePrompt();
  client.send({ type: "continue", role });
  // 仅准入弹窗额外发 gate_ok（与 G 键同效）
  if (!promptAllowSubject) {
    client.send({ type: "operator", action: "gate_ok" });
  }
  setStatus("已连接");
}

function sendOperator(action) {
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

let qOpen = false;

function showQuestionnaire(msg) {
  if (!el.qWrap || sessionDone) return;
  el.qTitle.textContent = msg.title || "问卷";
  el.qBody.innerHTML = "";
  el.qHint.textContent = "";
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
  setHelpTip("请完成后点击「提交问卷」");
}

function hideQuestionnaire() {
  if (!el.qWrap) return;
  el.qWrap.classList.add("hidden");
  el.qWrap.setAttribute("aria-hidden", "true");
  qOpen = false;
}

function submitQuestionnaire() {
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
  client.send({ type: "questionnaire_result", form: "post", answers });
  if (el.qHint) el.qHint.textContent = "已提交，感谢配合";
  setTimeout(hideQuestionnaire, 800);
}

if (el.qSubmit) {
  el.qSubmit.addEventListener("click", submitQuestionnaire);
}

if (maybeDemo()) {
  // v2 演示模式：渲染层兜底，不启动 ws
} else {
const client = new WsClient(
  wsUrl,
  (msg) => {
    if (msg.type === "v2_stage") { handleV2Stage(msg.stage, msg.ctx, msg.data); return; }
    if (msg.type === "hud") {
      if (el.text) el.text.textContent = msg.text || "";
      if (el.sub) el.sub.textContent = msg.subtext || "";
      if (el.cross) el.cross.classList.toggle("hidden", !msg.show_cross);
    } else if (msg.type === "stage") {
      // 流程已推进时收起残留弹窗（例如 G 已确认但前端未关）
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
      if (!msg.ok && msg.errors && msg.errors.length) {
        if (el.qHint) el.qHint.textContent = `提交未通过：${msg.errors.join("；")}`;
      }
    } else if (msg.type === "prompt") {
      showPrompt(msg);
    } else if (msg.type === "operator_state") {
      updateOpState(msg);
    } else if (msg.type === "session") {
      if (msg.status === "done") {
        sessionDone = true;
        hidePrompt();
        if (el.text) el.text.textContent = "本会话结束";
        if (el.sub) el.sub.textContent = "可以关闭页面";
        setOffline(false);
        setStatus("完成");
      } else if (msg.status === "error") {
        setStatus(`错误: ${msg.message || ""}`);
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
  el.promptBtn.addEventListener("click", () => {
    // 准入弹窗不允许被试：按钮按操作者确认
    sendContinue(promptAllowSubject ? "subject" : "operator");
  });
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
  client.send({ type: "sync" });
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
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
