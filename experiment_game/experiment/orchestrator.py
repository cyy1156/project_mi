"""操作台编排：空闲起服务 → 校验配置 → 开会话 → 等待诱导页 ready → SessionRunner。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_game.experiment.alignment import write_alignment_bundle
from experiment_game.experiment.defaults_store import (
    defaults_path,
    load_operator_defaults,
    save_operator_defaults,
)
from experiment_game.experiment.session_layout import finalize_session_layout
from experiment_game.acquisition import AcquisitionFacade, DEFAULT_CHANNEL_LABELS
from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.http_static import StaticServer
from experiment_game.experiment.markers import MarkerPublisher
from experiment_game.experiment.run_config import merge_run_config, validate_run_config
from experiment_game.experiment.serial_ports import list_serial_ports
from experiment_game.experiment.session import (
    SessionMeta,
    SessionPaths,
    create_session_dir,
    update_session_meta,
    write_session_meta,
)
from experiment_game.experiment.session_runner import Phase2Config, SessionRunner
from experiment_game.experiment.timing import timing_from_dict
from experiment_game.experiment.questionnaire import (
    latest_session_dir,
    post_form_payload,
    save_post_result,
    summarize_post_answers,
    validate_post_answers,
)
from experiment_game.experiment.local_ports import check_ports_free, format_port_conflict
from experiment_game.experiment.ws_bridge import WsBridge
from experiment_game.offline.phase4_service import run_phase4_for_session

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEB_ROOT = _PKG_ROOT / "web"


def _next_session_id(session_id: str, step: int) -> str:
    """换场继续段的 session_id：ses01 → ses02（保留补零）；无数字后缀则 _s2。"""
    m = re.match(r"^(.*?)(\d+)$", session_id)
    if m:
        width = len(m.group(2))
        return f"{m.group(1)}{int(m.group(2)) + step:0{width}d}"
    return f"{session_id}_s{step + 1}"


class OperatorService:
    """
    常驻 HTTP + WS；收到 session_start 后在工作线程跑一场实验。
    CLI 仍可用 run_phase2_session；本类专供 open_operator。
    """

    def __init__(
        self,
        *,
        http_port: int = 8080,
        ws_port: int = 8765,
        web_root: Optional[Path] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else _REPO_ROOT
        self.web_root = Path(web_root) if web_root else _WEB_ROOT
        self.http_port = http_port
        self.ws_port = ws_port
        self.bridge = WsBridge(port=ws_port)
        self.http = StaticServer(self.web_root, port=http_port)
        self._lock = threading.Lock()
        self._busy = False
        self._worker: Optional[threading.Thread] = None
        self._acq: Optional[AcquisitionFacade] = None
        self._events: Optional[EventLogger] = None
        self._markers: Optional[MarkerPublisher] = None
        self._paths: Optional[SessionPaths] = None
        self._last_config: Optional[Dict[str, Any]] = None
        self._stop_servers = threading.Event()
        # 换场（B）与问卷（Q）状态
        self._runner: Optional[SessionRunner] = None
        self._split_waiting = False
        self._q_context: Optional[Dict[str, Any]] = None

    @property
    def operator_url(self) -> str:
        return f"http://127.0.0.1:{self.http_port}/operator.html#setup"

    @property
    def subject_url(self) -> str:
        return (
            f"http://127.0.0.1:{self.http_port}/"
            f"?ws=ws://127.0.0.1:{self.ws_port}"
        )

    def start(self) -> None:
        busy = check_ports_free(
            [
                ("127.0.0.1", self.http_port),
                ("127.0.0.1", self.ws_port),
            ]
        )
        if busy:
            raise RuntimeError(
                format_port_conflict(busy, operator_url=self.operator_url)
            )
        self.bridge.set_on_message(self._on_ws_message)
        self.bridge.start()
        self.http.start()
        self._emit_acq_status("idle", "等待 Setup 开始实验")
        print(f"操作台: {self.operator_url}")
        print(f"诱导页: {self.subject_url}")
        print(f"WebSocket: {self.bridge.url}")

    def stop(self) -> None:
        self._stop_servers.set()
        self._shutdown_session_resources()
        try:
            self.http.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.bridge.stop()
        except Exception:  # noqa: BLE001
            pass

    def serve_forever(self) -> int:
        self.start()
        try:
            while not self._stop_servers.is_set():
                time.sleep(0.5)
                if self._worker is not None and not self._worker.is_alive():
                    self._worker = None
        except KeyboardInterrupt:
            print("\n用户中断", file=sys.stderr)
            return 130
        finally:
            self.stop()
        return 0

    def _on_ws_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "config_validate":
            cfg, errors = validate_run_config(
                msg.get("run_config") or {},
                repo_root=self.repo_root,
            )
            self.bridge.broadcast(
                {
                    "type": "config_ack",
                    "ok": not errors,
                    "errors": errors,
                    "run_config": cfg if not errors else None,
                }
            )
        elif mtype == "session_start":
            self._handle_session_start(msg.get("run_config") or {})
        elif mtype == "open_folder":
            self._open_folder(str(msg.get("path") or ""))
        elif mtype == "open_subject_page":
            self._open_subject_page()
        elif mtype == "list_serial_ports":
            self._list_serial_ports()
        elif mtype == "save_defaults":
            self._save_defaults(msg.get("run_config") or {})
        elif mtype == "run_phase4":
            self._handle_run_phase4(str(msg.get("path") or ""), msg)
        elif mtype == "operator" and str(msg.get("action") or "") == "split_session":
            self._handle_split_request()
        elif mtype == "questionnaire_open":
            self._handle_questionnaire_open()
        elif mtype == "questionnaire_result":
            self._handle_questionnaire_result(msg)
        elif mtype == "client_stats":
            self._handle_client_stats(msg)
        elif mtype == "operator_hello":
            file_defaults, warn = load_operator_defaults(
                defaults_path(repo_pkg=_PKG_ROOT),
                repo_root=self.repo_root,
            )
            self.bridge.broadcast(
                {
                    "type": "operator_hello",
                    "message": "operator_connected",
                    "operator_url": self.operator_url,
                    "subject_url": self.subject_url,
                    "defaults": file_defaults,
                    "builtin_defaults": merge_run_config(None),
                    "defaults_path": str(defaults_path(repo_pkg=_PKG_ROOT)),
                    "defaults_warning": warn,
                    "serial_ports": list_serial_ports(),
                }
            )

    def _handle_split_request(self) -> None:
        """操作台 B 键：会话中 = 请求换场；换场等待中 = 开始下一段。"""
        if self._split_waiting:
            self.bridge.set_event("split_request")
            return
        runner = self._runner
        if self._busy and runner is not None:
            runner.request_split()
            return
        self.bridge.broadcast(
            {
                "type": "operator_hint",
                "message": "B 换场需要会话进行中（正式采集段）或换场等待中",
            }
        )

    def _handle_questionnaire_open(self) -> None:
        """操作台 Q 键：把后测问卷推到诱导页（不自动注入任何 session）。"""
        target = None
        subject_id = ""
        session_id = ""
        if self._paths is not None and self._paths.root.is_dir():
            target = self._paths.root
        else:
            cfg = self._last_config or {}
            save_root = (cfg.get("storage") or {}).get("save_root")
            subject_id = str((cfg.get("subject") or {}).get("subject_id") or "")
            if save_root:
                found = latest_session_dir(Path(save_root), subject_id)
                if found is not None:
                    target = found
        if target is not None:
            parts = target.name.rsplit("_", 2)
            if len(parts) == 3:
                subject_id, session_id = parts[0], parts[1]
        if target is None:
            self.bridge.broadcast(
                {
                    "type": "questionnaire_ack",
                    "ok": False,
                    "message": "未找到可关联的会话目录（先完成一场采集）",
                }
            )
            return
        self._q_context = {
            "session_root": str(target),
            "subject_id": subject_id,
            "session_id": session_id,
        }
        payload = post_form_payload()
        payload["session_root"] = str(target)
        self.bridge.broadcast(payload)
        print(f"[operator] 问卷已推送到诱导页（关联 {target.name}）")

    def _handle_questionnaire_result(self, msg: Dict[str, Any]) -> None:
        errors = validate_post_answers(msg.get("answers"))
        if errors:
            self.bridge.broadcast(
                {"type": "questionnaire_ack", "ok": False, "errors": errors}
            )
            return
        answers = dict(msg.get("answers") or {})
        ctx = self._q_context or {}
        target = Path(ctx.get("session_root") or "")
        if not target.is_dir():
            # 页面刷新等导致上下文丢失：退回最新会话目录
            cfg = self._last_config or {}
            save_root = (cfg.get("storage") or {}).get("save_root")
            subject_id = str((cfg.get("subject") or {}).get("subject_id") or "")
            target = (
                latest_session_dir(Path(save_root), subject_id)
                if save_root
                else None
            ) or target
        try:
            path = save_post_result(
                answers,
                session_root=target,
                subject_id=str(ctx.get("subject_id") or ""),
                session_id=str(ctx.get("session_id") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {"type": "questionnaire_ack", "ok": False, "message": str(exc)}
            )
            return
        if self._events is not None:
            try:
                self._events.emit("questionnaire_done", form="post", path=str(path))
            except Exception:  # noqa: BLE001
                pass
        self.bridge.clear_pending_questionnaire()
        self._q_context = None
        self.bridge.broadcast(
            {
                "type": "questionnaire_ack",
                "ok": True,
                "summary": summarize_post_answers(answers),
                "path": str(path),
            }
        )
        print(f"[operator] 问卷已保存: {path}")

    def _handle_client_stats(self, msg: Dict[str, Any]) -> None:
        """诱导页渲染遥测：写入当前会话 events.jsonl（无会话时丢弃）。"""
        events = self._events
        if events is None:
            return
        try:
            events.emit(
                "client_stats",
                fps=msg.get("fps"),
                max_gap_ms=msg.get("max_gap_ms"),
            )
        except Exception:  # noqa: BLE001
            pass

    def _handle_session_start(self, raw: Dict[str, Any]) -> None:
        with self._lock:
            if self._busy:
                self.bridge.broadcast(
                    {
                        "type": "config_ack",
                        "ok": False,
                        "errors": ["已有会话在进行，请先结束或中止"],
                    }
                )
                return
            cfg, errors = validate_run_config(raw, repo_root=self.repo_root)
            if errors:
                self.bridge.broadcast(
                    {"type": "config_ack", "ok": False, "errors": errors}
                )
                return
            self._busy = True
            self._last_config = cfg

        self.bridge.broadcast(
            {
                "type": "config_ack",
                "ok": True,
                "errors": [],
                "run_config": cfg,
                "starting": True,
            }
        )
        self._worker = threading.Thread(
            target=self._run_session_safe,
            args=(cfg,),
            name="operator-session",
            daemon=True,
        )
        self._worker.start()

    def _run_session_safe(self, cfg: Dict[str, Any]) -> None:
        try:
            self._run_session(cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[operator] 会话错误: {exc}", file=sys.stderr)
            self.bridge.broadcast(
                {"type": "session", "status": "error", "message": str(exc)}
            )
            self._emit_acq_status("error", str(exc))
            if self._paths is not None:
                files = self._list_session_files(self._paths.root)
                self.bridge.broadcast(
                    {
                        "type": "session_saved",
                        "root": str(self._paths.root),
                        "files": files,
                        "acq_enabled": bool(
                            (cfg.get("acquisition") or {}).get("enabled")
                        ),
                        "train_eligible": False,
                        "message": f"异常结束: {exc}",
                    }
                )
        finally:
            self._shutdown_session_resources()
            with self._lock:
                self._busy = False

    def _run_session(self, cfg: Dict[str, Any]) -> None:
        # 重置桥接事件，避免上一场 ready/abort 残留
        for name in ("ready", "continue", "abort", "gate_ok", "split_request"):
            self.bridge.clear_event(name)
        self.bridge.paused = False
        self.bridge.reject_requested = False
        self._runner = None
        self._split_waiting = False

        sub = cfg["subject"]
        acq_cfg = cfg["acquisition"]
        exp = cfg["experiment"]
        storage = cfg["storage"]
        save_root = Path(storage["save_root"])
        timing = timing_from_dict(exp.get("timing"))
        trials_total = int(exp["acquire_trials"])
        settle_s = float((exp.get("split") or {}).get("settle_s", 15.0))

        subject_id = sub["subject_id"]
        trials_done_total = 0
        segment = 0
        while True:
            segment += 1
            is_first = segment == 1
            remaining = trials_total - trials_done_total
            session_id = (
                sub["session_id"] if is_first
                else _next_session_id(sub["session_id"], segment - 1)
            )

            paths = create_session_dir(save_root, subject_id, session_id)
            self._paths = paths

            # 会话快照（便于复现；UI-2 正式化）。继续段记录该段实际 trial 数
            seg_cfg = dict(cfg)
            seg_cfg = {
                **cfg,
                "subject": {**sub, "session_id": session_id},
                "experiment": {**exp, "acquire_trials": remaining},
            }
            (paths.root / "run_config.json").write_text(
                json.dumps(seg_cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if is_first:
                self._maybe_persist_last_config(cfg)

            use_synthetic = acq_cfg["board_mode"] != "cyton"
            acq_on = bool(acq_cfg["enabled"])
            meta = SessionMeta(
                subject_id=subject_id,
                session_id=session_id,
                phase_mode="phase2_full",
                use_synthetic=use_synthetic if acq_on else True,
                trial_count=remaining,
                object="cup",
                scene="home_desk",
                notes=str(sub.get("notes") or "operator_console")
                + ("" if is_first else "；换场继续段"),
                channel_labels=list(
                    acq_cfg.get("channel_labels") or DEFAULT_CHANNEL_LABELS
                ),
            )
            write_session_meta(paths.meta_json, meta)
            # 时序快照写入 meta：每个会话可追溯当场的范式时长构成
            update_session_meta(
                paths.meta_json,
                timing=timing.to_dict(),
                trial_total_s=round(timing.total_s, 2),
                segment=segment,
            )

            self.bridge.broadcast(
                {
                    "type": "session_started",
                    "session_root": str(paths.root),
                    "subject_url": self.subject_url,
                    "acq_enabled": acq_on,
                    "board_mode": acq_cfg["board_mode"],
                    "serial_port": acq_cfg.get("serial_port"),
                    "acquire_trials": remaining,
                    "timing": timing.to_dict(),
                    "trial_total_s": round(timing.total_s, 2),
                    "save_root": str(save_root),
                    "open_subject_page": exp.get("open_subject_page", True),
                    "segment": segment,
                }
            )

            events = EventLogger(paths.events_jsonl)
            self._events = events
            markers = MarkerPublisher(enabled=acq_on and bool(acq_cfg.get("markers_lsl", True)))
            self._markers = markers

            phase_cfg = Phase2Config(
                acquire_trials=remaining,
                learn_trials_per_step=int(exp["learn_trials_per_step"]),
                seed=exp.get("seed"),
                skip_adapt=bool(exp.get("skip_adapt", False)) or not is_first,
                skip_learn=bool(exp.get("skip_learn", False)) or not is_first,
                skip_gate=bool(exp.get("skip_gate", False)) or not is_first,
                auto_continue=False,
                rotate_objects=True,
                rotate_scenes=True,
                settle_s=0.0 if is_first else settle_s,
            )
            runner = SessionRunner(events, markers, self.bridge, timing=timing, config=phase_cfg)
            self._runner = runner

            if acq_on:
                self._emit_acq_status("connecting", "正在启动采集…")
                try:
                    filt = acq_cfg.get("filter") or {}
                    self._acq = AcquisitionFacade(
                        use_synthetic=use_synthetic,
                        serial_port=str(acq_cfg.get("serial_port") or "COM5"),
                        channel_labels=meta.channel_labels,
                        filter_enabled=bool(filt.get("enabled", True)),
                        bandpass_low_hz=float(filt.get("bandpass_low_hz", 0.5)),
                        bandpass_high_hz=float(filt.get("bandpass_high_hz", 45.0)),
                        notch_low_hz=float(filt.get("notch_low_hz", 49.0)),
                        notch_high_hz=float(filt.get("notch_high_hz", 51.0)),
                    )
                    self._acq.create()
                    self._acq.start(paths.eeg_csv)
                    time.sleep(1.5)
                    self._emit_acq_status("recording", "录制中")
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    if not use_synthetic:
                        msg = (
                            f"{msg}\n真机排查：关闭 OpenBCI GUI 串口直播；"
                            f"确认设备管理器串口为 {acq_cfg.get('serial_port')}；重新插拔 USB。"
                        )
                    self._emit_acq_status("error", msg)
                    raise RuntimeError(msg) from exc
            else:
                self._emit_acq_status("idle", "本次未开启采集")

            events.emit(
                "session_start",
                subject_id=subject_id,
                session_id=session_id,
                phase="phase2",
                segment=segment,
            )
            markers.push(
                f"session_start|subject={subject_id}|session={session_id}|phase=phase2"
            )

            self.bridge.broadcast({"type": "session", "status": "running", "phase": "waiting_ready"})

            if is_first and exp.get("open_subject_page", True):
                # 后端再开一次，作为弹窗拦截时的兜底（与前端 window.open 并存）
                try:
                    webbrowser.open(self.subject_url)
                except Exception as exc:  # noqa: BLE001
                    print(f"[operator] 打开诱导页失败: {exc}", file=sys.stderr)

            if is_first:
                timeout = float(exp.get("ready_timeout_s") or 60)
                print(f"[operator] 等待诱导页 ready（{timeout:.0f}s）…")
                try:
                    runner.wait_browser_ready(timeout=timeout)
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"诱导页未在 {timeout:.0f}s 内 ready；请点「重新打开诱导页」并允许弹窗"
                    ) from exc

            runner.run_all()
            self._runner = None

            events.emit(
                "session_end",
                subject_id=subject_id,
                session_id=session_id,
                phase="phase2",
            )
            markers.push("session_end|phase=phase2")

            # 先停录制，再整理 continuous / by_phase / alignment
            self._shutdown_session_resources()

            layout = str(storage.get("save_layout") or "phase_folders")
            try:
                finalize_session_layout(
                    paths.root,
                    save_layout=layout,
                    save_continuous=bool(storage.get("save_continuous_master", True)),
                    save_phase_slices=bool(
                        storage.get("save_phase_slices") or layout == "phase_folders"
                    ),
                    acq_enabled=acq_on,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[operator] 整理落盘目录失败: {exc}", file=sys.stderr)

            verify = {}
            try:
                verify = write_alignment_bundle(paths.root, acq_enabled=acq_on)
            except Exception as exc:  # noqa: BLE001
                print(f"[operator] alignment 失败: {exc}", file=sys.stderr)
                verify = {"passed": False, "errors": [str(exc)]}

            phase4_result: Optional[Dict[str, Any]] = None
            if acq_on and bool(storage.get("auto_phase4")):
                p4 = exp.get("phase4") if isinstance(exp.get("phase4"), dict) else {}
                print(
                    "[operator] auto_phase4：仅 acquire + 未 reject "
                    f"({p4.get('window_mode', 'fixed')} w{p4.get('win_sec', 2.0):g}s)…"
                )
                phase4_result = run_phase4_for_session(
                    paths.root,
                    repo_root=self.repo_root,
                    window_mode=str(p4.get("window_mode") or "fixed"),
                    win_sec=float(p4.get("win_sec", 2.0)),
                    hop_ms=float(p4.get("hop_ms", 100.0)),
                )
                print(
                    f"[operator] Phase4: ok={phase4_result.get('ok')} "
                    f"{phase4_result.get('message')} → {phase4_result.get('epochs_dir')}"
                )

            files = self._list_session_files(paths.root)
            trials_done_total += runner.trials_done
            split_continue = (
                runner.split_requested
                and (trials_total - trials_done_total) > 0
            )

            if split_continue:
                # 换场中场：本场已保存；等操作者第二次 B 开新 session
                self.bridge.broadcast(
                    {
                        "type": "session_segment_saved",
                        "root": str(paths.root),
                        "files": files,
                        "segment": segment,
                        "trials_done": runner.trials_done,
                        "trials_remaining": trials_total - trials_done_total,
                        "message": (
                            f"第 {segment} 段已保存（{runner.trials_done} trial）；"
                            f"剩余 {trials_total - trials_done_total} 个 trial。"
                            "操作者引导抬手休息后，再按 B 开始下一段（Esc 放弃剩余）"
                        ),
                    }
                )
                self.bridge.broadcast(
                    {
                        "type": "prompt",
                        "id": "session_split",
                        "title": "中场休息",
                        "body": (
                            "请按操作者引导抬起双手、对照画面休息。"
                            "坐好后保持放松，等待操作者开始下一场。"
                        ),
                        "button": "等待操作者开始下一场",
                        "allow_subject": False,
                    }
                )
                print(
                    f"[operator] 换场：第 {segment} 段完成（{runner.trials_done} trial），"
                    f"等待第二次 B…"
                )
                self._split_waiting = True
                resumed = self._wait_split_or_abort(timeout_s=7200.0)
                self._split_waiting = False
                self.bridge.clear_event("split_request")
                self.bridge.clear_pending_prompt()
                if resumed:
                    print("[operator] 第二次 B：开新 session 继续剩余 trial…")
                    continue
                # 等待中被中止/超时：按正常结束处理（本段数据已保存）
                self.bridge.broadcast(
                    {
                        "type": "session_saved",
                        "root": str(paths.root),
                        "files": files,
                        "acq_enabled": acq_on,
                        "train_eligible": bool(acq_on and verify.get("passed", False)),
                        "verify": verify,
                        "phase4": phase4_result,
                        "message": "换场等待被中止/超时；已保留本段数据",
                    }
                )
                self.bridge.broadcast({"type": "session", "status": "done"})
                if acq_on:
                    self._emit_acq_status("stopped", "录制已停止")
                return

            self.bridge.broadcast(
                {
                    "type": "session_saved",
                    "root": str(paths.root),
                    "files": files,
                    "acq_enabled": acq_on,
                    "train_eligible": bool(acq_on and verify.get("passed", False)),
                    "verify": verify,
                    "phase4": phase4_result,
                    "segments": segment,
                    "trials_done": trials_done_total,
                    "message": "会话已结束" if acq_on else "会话已结束（无 EEG，不可训练切窗）",
                }
            )
            self.bridge.broadcast({"type": "session", "status": "done"})
            if acq_on:
                self._emit_acq_status("stopped", "录制已停止")
            print(f"[operator] 会话目录: {paths.root}")
            return

    def _wait_split_or_abort(self, timeout_s: float) -> bool:
        """换场等待：第二次 B 返回 True；中止/超时返回 False。"""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.bridge.should_abort():
                print("[operator] 换场等待中被中止（Esc）")
                return False
            if self.bridge.wait_client_event("split_request", timeout=0.5):
                return True
        return False

    def _handle_run_phase4(self, path: str, msg: Optional[Dict[str, Any]] = None) -> None:
        target = Path(path).expanduser() if path else (self._paths.root if self._paths else None)
        if target is None or not Path(target).is_dir():
            self.bridge.broadcast(
                {
                    "type": "phase4_ack",
                    "ok": False,
                    "message": "无效会话路径",
                    "path": path,
                }
            )
            return

        # 切窗参数（Summary 页控件可带；未带则用默认 fixed 2s）
        p4 = (msg or {}).get("phase4") if isinstance((msg or {}).get("phase4"), dict) else {}
        mode = str(p4.get("window_mode") or "fixed").lower()
        if mode not in ("fixed", "slide"):
            mode = "fixed"
        try:
            win_sec = float(p4.get("win_sec", 2.0))
        except (TypeError, ValueError):
            win_sec = 2.0
        try:
            hop_ms = float(p4.get("hop_ms", 100.0))
        except (TypeError, ValueError):
            hop_ms = 100.0

        def _job() -> None:
            try:
                result = run_phase4_for_session(
                    Path(target),
                    repo_root=self.repo_root,
                    window_mode=mode,
                    win_sec=win_sec,
                    hop_ms=hop_ms,
                )
                self.bridge.broadcast({"type": "phase4_ack", **result, "path": str(target)})
            except Exception as exc:  # noqa: BLE001
                self.bridge.broadcast(
                    {
                        "type": "phase4_ack",
                        "ok": False,
                        "message": str(exc),
                        "path": str(target),
                        "summary": {},
                    }
                )

        threading.Thread(target=_job, name="phase4", daemon=True).start()

    def _shutdown_session_resources(self) -> None:
        acq = self._acq
        self._acq = None
        if acq is not None:
            try:
                report = acq.stop()
                print(f"[operator] 录制停止: {report.get('message')}")
            except Exception as exc:  # noqa: BLE001
                print(f"[operator] 停止录制异常: {exc}", file=sys.stderr)
            try:
                acq.shutdown()
            except Exception:  # noqa: BLE001
                pass

        if self._events is not None:
            try:
                self._events.close()
            except Exception:  # noqa: BLE001
                pass
            self._events = None

        if self._markers is not None:
            try:
                self._markers.close()
            except Exception:  # noqa: BLE001
                pass
            self._markers = None

        if self._paths is not None:
            try:
                update_session_meta(
                    self._paths.meta_json, session_dir=str(self._paths.root)
                )
            except Exception:  # noqa: BLE001
                pass

    def _emit_acq_status(self, state: str, message: str = "") -> None:
        self.bridge.broadcast(
            {"type": "acq_status", "state": state, "message": message}
        )

    def _list_serial_ports(self) -> None:
        try:
            ports = list_serial_ports()
            self.bridge.broadcast(
                {"type": "serial_ports", "ok": True, "ports": ports}
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {
                    "type": "serial_ports",
                    "ok": False,
                    "ports": [],
                    "message": str(exc),
                }
            )

    def _save_defaults(self, raw: Dict[str, Any]) -> None:
        ok, message, cfg = save_operator_defaults(
            raw,
            defaults_path(repo_pkg=_PKG_ROOT),
            repo_root=self.repo_root,
        )
        self.bridge.broadcast(
            {
                "type": "save_defaults_ack",
                "ok": ok,
                "message": message,
                "run_config": cfg,
                "path": message if ok else str(defaults_path(repo_pkg=_PKG_ROOT)),
            }
        )

    def _maybe_persist_last_config(self, cfg: Dict[str, Any]) -> None:
        ui = cfg.get("ui") or {}
        if not ui.get("remember_last_config", True):
            return
        ok, msg, _ = save_operator_defaults(
            cfg,
            defaults_path(repo_pkg=_PKG_ROOT),
            repo_root=self.repo_root,
        )
        if ok:
            print(f"[operator] 已更新本地默认配置: {msg}")
        else:
            print(f"[operator] 更新默认配置失败: {msg}", file=sys.stderr)

    def _open_subject_page(self) -> None:
        try:
            webbrowser.open(self.subject_url)
            self.bridge.broadcast(
                {
                    "type": "subject_page_opened",
                    "url": self.subject_url,
                    "ok": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {
                    "type": "subject_page_opened",
                    "url": self.subject_url,
                    "ok": False,
                    "message": str(exc),
                }
            )

    def _open_folder(self, path: str) -> None:
        target = Path(path).expanduser()
        if not target.exists():
            self.bridge.broadcast(
                {
                    "type": "open_folder_ack",
                    "ok": False,
                    "path": path,
                    "message": "路径不存在",
                }
            )
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
            self.bridge.broadcast(
                {"type": "open_folder_ack", "ok": True, "path": str(target)}
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {
                    "type": "open_folder_ack",
                    "ok": False,
                    "path": str(target),
                    "message": str(exc),
                }
            )

    @staticmethod
    def _list_session_files(root: Path) -> List[str]:
        out: List[str] = []
        if not root.is_dir():
            return out
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(root)).replace("\\", "/"))
        return out
