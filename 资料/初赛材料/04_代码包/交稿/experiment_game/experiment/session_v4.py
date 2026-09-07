"""v4 实验前数据质量检测会话（无模型、无 trial）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.inference_v2 import FS, RingBuffer
from experiment_game.experiment.markers import MarkerPublisher
from experiment_game.experiment.signal_quality import (
    diagnose_eeg_window,
    live_channel_stats,
    summarize_v4_session,
)
from experiment_game.experiment.trial_sm import SessionAbort, wait_until
from experiment_game.experiment.v4_config import V4Config
from experiment_game.experiment.v4_quality import V4QualityMonitor
from experiment_game.experiment.v4_report import write_v4_report
from experiment_game.experiment.ws_bridge import WsBridge


def attach_v4_lsl(
    cfg: V4Config,
    *,
    lsl_timeout_s: Optional[float] = None,
) -> RingBuffer:
    buf = RingBuffer()
    buf.attach_lsl(cfg.lsl_stream_name, timeout_s=lsl_timeout_s or cfg.lsl_timeout_s)
    return buf


def diagnose_v4_lsl(
    cfg: V4Config,
    *,
    lsl_timeout_s: Optional[float] = None,
    on_console: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[RingBuffer], List[str]]:
    log = on_console or (lambda _m: None)
    try:
        buf = attach_v4_lsl(cfg, lsl_timeout_s=lsl_timeout_s)
        log(f"[v4] LSL 流 {cfg.lsl_stream_name} 已挂接")
        return buf, []
    except Exception as exc:  # noqa: BLE001
        return None, [f"LSL {cfg.lsl_stream_name} 不可用: {exc}"]


def run_v4_session(
    events: EventLogger,
    markers: MarkerPublisher,
    bridge: WsBridge,
    buf: RingBuffer,
    *,
    on_console: Callable[[str], None] = print,
    config_path: Optional[str] = None,
    v4_overrides: Optional[Dict[str, Any]] = None,
    session_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from pylsl import local_clock

    cfg = V4Config.load_yaml(config_path) if config_path else V4Config.load_yaml()
    ignored = cfg.apply_overrides(v4_overrides)
    if ignored:
        on_console(f"[v4] 忽略未知 overrides: {ignored}")
    verr = cfg.verify_errors()
    if verr:
        raise ValueError("v4 配置无效: " + "; ".join(verr))

    sq = cfg.signal_quality_config()
    names = list(cfg.channel_labels)
    monitor = V4QualityMonitor(cfg)
    win_n = max(1, int(round(cfg.eval_window_s * FS)))
    session_path = session_dir or (Path(events.path).parent if hasattr(events, "path") else Path("."))

    bridge.broadcast(
        {
            "type": "v4_start",
            "duration_s": cfg.duration_s,
            "pass_streak_required": cfg.pass_streak_required,
            "channel_labels": names,
            "eval_interval_s": cfg.eval_interval_s,
        }
    )
    bridge.broadcast(
        {
            "type": "hud",
            "text": "数据质量检测",
            "subtext": "达标：松手静坐，等格子变绿（每 3s 评估）。触诊：极轻按，看哪格数字跳",
            "show_cross": True,
        }
    )
    events.emit("v4_start", phase="v4", duration_s=cfg.duration_s)
    markers.push("v4_start")

    from experiment_game.experiment.session_base import SessionServices, attach_eeg_health
    from experiment_game.runtime.eeg_bus import resolve_eeg_watchdog

    eeg_abort_s = float(resolve_eeg_watchdog()["abort_s"])
    _eeg_health = attach_eeg_health(
        buf,
        SessionServices(events, markers, bridge, on_console),
        tag="v4",
        enabled=True,
    )

    t_start = local_clock()
    t_end = t_start + cfg.duration_s
    live_interval = 0.5
    live_win_s = 0.5
    live_n = max(1, int(round(live_win_s * FS)))
    next_eval = t_start + cfg.eval_interval_s
    next_live = t_start + live_interval
    on_console(
        f"[v4] 检测开始：最长 {cfg.duration_s:.0f}s，"
        f"QC 每 {cfg.eval_interval_s:.0f}s · 通道刷新 {live_interval:.1f}s"
    )

    try:
        while local_clock() < t_end:
            if bridge.should_abort():
                raise SessionAbort("operator_abort")
            if _eeg_health is not None:
                _eeg_health.tick(buf)
            st = buf.stale_status(eeg_abort_s)
            if st is not None:
                age = float(st["age_s"])
                msg = (
                    f"EEG 断流：已 {age:.1f}s 无新样本（阈值 {eeg_abort_s:.0f}s）。"
                    "请检查 dongle/COM/USB。"
                )
                on_console(f"[v4] ERR {msg}")
                events.emit("eeg_stale", phase="v4", age_s=age, timeout_s=st["timeout_s"])
                bridge.broadcast(
                    {
                        "type": "eeg_stale",
                        "age_s": age,
                        "timeout_s": float(st["timeout_s"]),
                        "n_samples": int(st["n_samples"]),
                        "message": msg,
                    }
                )
                bridge.broadcast({"type": "acq_status", "state": "error", "message": msg})
                raise SessionAbort(f"eeg_stale:{age:.1f}s")
            now = local_clock()

            if now >= next_live:
                raw_live = buf.snapshot_tail(live_win_s, t_now_lsl=now)
                if raw_live is not None and raw_live.shape[0] >= live_n:
                    bridge.broadcast(
                        {
                            "type": "v4_live",
                            "ts": now,
                            **live_channel_stats(raw_live, names),
                        }
                    )
                next_live += live_interval

            if now >= next_eval:
                raw = buf.snapshot_tail(cfg.eval_window_s, t_now_lsl=now)
                elapsed = now - t_start
                if raw is None or raw.shape[0] < win_n:
                    diag = {
                        "window_ok": False,
                        "window_reason": "buffer_warmup",
                        "metrics": {},
                        "per_channel": [],
                        "per_channel_ok": [False] * 8,
                        "problems": [
                            {
                                "channel": "",
                                "reason": "buffer_warmup",
                                "detail": "缓冲预热中",
                                "hint": "请等待数秒让数据填满",
                            }
                        ],
                    }
                else:
                    diag = diagnose_eeg_window(raw, sq, channel_names=names)

                pass_evt = monitor.update(diag, elapsed_s=elapsed)
                payload = {
                    "type": "v4_quality",
                    "ts": now,
                    **diag,
                    "window_idx": monitor.window_idx,
                    "elapsed_s": round(elapsed, 2),
                    "pass_streak": monitor.streak,
                    "rolling_verdict": monitor.rolling_verdict(),
                }
                bridge.broadcast(payload)

                if pass_evt:
                    bridge.broadcast({"type": "v4_pass", **pass_evt})
                    events.emit("v4_pass", phase="v4", **pass_evt)
                    on_console(f"[v4] ✅ {pass_evt['message']}")
                    if cfg.auto_stop_on_pass:
                        break

                next_eval += cfg.eval_interval_s

            wait_until(
                min(next_live, next_eval, t_end),
                is_paused=bridge.is_paused,
                should_abort=bridge.should_abort,
            )
    except SessionAbort as exc:
        reason = getattr(exc, "reason", None) or str(exc) or "operator_abort"
        on_console(f"[v4] 会话中止 · {reason}")
        events.emit("v4_abort", phase="v4", reason=reason)
        if str(reason).startswith("eeg_stale"):
            bridge.broadcast(
                {
                    "type": "session",
                    "status": "error",
                    "message": "EEG 断流，v4 已中止（请检查 dongle/COM）",
                    "phase": "v4_session",
                }
            )

    duration = local_clock() - t_start
    summary = summarize_v4_session(
        monitor.history,
        duration_s=duration,
        pass_streak_required=cfg.pass_streak_required,
        achieved_stable=monitor.achieved_stable,
        time_to_stable_s=monitor.time_to_stable_s,
        channel_names=names,
        unused_channels=list(cfg.unused_channels),
        scoring_channels=list(cfg.scoring_channels),
    )
    write_v4_report(session_path, summary, history=monitor.history)
    events.emit("v4_end", phase="v4", **summary)
    markers.push(f"v4_end|verdict={summary.get('verdict')}")
    bridge.broadcast({"type": "v4_summary", **summary})
    on_console(f"[v4] 结束：{summary.get('verdict')} — {summary.get('recommendation')}")
    return summary
