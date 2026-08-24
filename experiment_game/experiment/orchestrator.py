"""操作台编排：空闲起服务 → 校验配置 → 开会话 → 等待诱导页 ready → SessionRunner。"""

from __future__ import annotations

import json
import os
import re
import socket
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


def _lan_ipv4_addrs() -> List[str]:
    """本机局域网 IPv4（排除 loopback），供监控端抄地址。"""
    found: List[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except OSError:
        pass
    if not found:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                found.append(ip)
        except OSError:
            pass
    return found


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
        serve_host: str = "127.0.0.1",
        web_root: Optional[Path] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else _REPO_ROOT
        self.web_root = Path(web_root) if web_root else _WEB_ROOT
        self.http_port = http_port
        self.ws_port = ws_port
        self.serve_host = serve_host
        self.bridge = WsBridge(host=serve_host, port=ws_port)
        self.http = StaticServer(self.web_root, host=serve_host, port=http_port)
        self._lock = threading.Lock()
        self._busy = False
        self._worker: Optional[threading.Thread] = None
        self._acq: Optional[AcquisitionFacade] = None
        self._last_acq_quality: Dict[str, Any] = {}
        self._events: Optional[EventLogger] = None
        self._markers: Optional[MarkerPublisher] = None
        self._paths: Optional[SessionPaths] = None
        self._last_config: Optional[Dict[str, Any]] = None
        self._stop_servers = threading.Event()
        # 换场（B）与问卷（Q）状态
        self._runner: Optional[SessionRunner] = None
        self._split_waiting = False
        self._q_context: Optional[Dict[str, Any]] = None
        # Cyton 链路监控（F3）
        self._link_monitor_stop = threading.Event()
        self._link_monitor_thread: Optional[threading.Thread] = None
        self._link_prev_pushed = 0
        self._link_firmware = ""

    @property
    def operator_url(self) -> str:
        # 本机浏览器仍用 127.0.0.1；LAN 地址在 start() 额外打印
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
        print(f"WebSocket: ws://127.0.0.1:{self.ws_port}")
        if self.serve_host not in ("127.0.0.1", "localhost"):
            print(f"绑定地址: {self.serve_host}:{self.http_port} / WS :{self.ws_port}")
            lan = _lan_ipv4_addrs()
            if lan:
                for ip in lan:
                    print(f"监控端打开: http://{ip}:{self.http_port}/operator.html#setup")
                    print(f"  （WS 自动连 ws://{ip}:{self.ws_port}）")
            else:
                print("未解析到局域网 IP；请在实验机执行 ipconfig 后告知监控端。")

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
            self.bridge.broadcast(
                {
                    "type": "operator_hint",
                    "message": "问卷失败：未找到会话目录（先完成一场采集）",
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
        save_hint = f"{target}/99_summary/questionnaire_post_*.json"
        self.bridge.broadcast(
            {
                "type": "operator_hint",
                "message": (
                    f"问卷已推送到诱导页（请保持诱导页打开）。"
                    f"提交后保存到：{save_hint}"
                ),
            }
        )
        print(f"[operator] 问卷已推送到诱导页（关联 {target.name}）→ {save_hint}")

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
        self.bridge.broadcast(
            {
                "type": "operator_hint",
                "message": f"问卷已保存：{path}",
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
                        "acq_quality": dict(self._last_acq_quality),
                    }
                )
        finally:
            self._shutdown_session_resources()
            with self._lock:
                self._busy = False

    def _run_session(self, cfg: Dict[str, Any]) -> None:
        # 重置桥接事件，避免上一场 ready/abort 残留
        for name in ("ready", "continue", "abort", "gate_ok", "split_request", "v2_guidance_confirm"):
            self.bridge.clear_event(name)
        self.bridge.paused = False
        self.bridge.reject_requested = False
        self._runner = None
        self._split_waiting = False
        self._q_context: Optional[Dict[str, Any]] = None
        self._last_acq_quality = {}

        exp = cfg["experiment"]
        phase_mode = str(exp.get("phase_mode") or "phase2_full")
        if phase_mode == "v3_session":
            self._run_v3_session(cfg)
            return
        if phase_mode == "v4_session":
            self._run_v4_session(cfg)
            return
        if phase_mode == "v2_session":
            self._run_v2_session(cfg)
            return

        sub = cfg["subject"]
        acq_cfg = cfg["acquisition"]
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
                try:
                    self._start_acquisition_pipeline(
                        paths, meta, acq_cfg, use_synthetic
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
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

            # 诱导页由操作台前端 window.open；弹窗被拦时前端会发 open_subject_page
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
                        "acq_quality": self._acq_quality_for_saved(verify),
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
                    "acq_quality": self._acq_quality_for_saved(verify),
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

    def _run_v2_session(self, cfg: Dict[str, Any]) -> None:
        """v2 会话模式：标定→准入→游戏；参数见 config/v2_session.yaml。"""
        sub = cfg["subject"]
        acq_cfg = cfg["acquisition"]
        exp = cfg["experiment"]
        storage = cfg["storage"]
        save_root = Path(storage["save_root"])
        subject_id = sub["subject_id"]
        session_id = sub["session_id"]

        paths = create_session_dir(save_root, subject_id, session_id)
        self._paths = paths
        (paths.root / "run_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._maybe_persist_last_config(cfg)

        use_synthetic = acq_cfg["board_mode"] != "cyton"
        acq_on = bool(acq_cfg["enabled"])
        meta = SessionMeta(
            subject_id=subject_id,
            session_id=session_id,
            phase_mode="v2_session",
            use_synthetic=use_synthetic if acq_on else True,
            trial_count=0,
            object="cup",
            scene="home_desk",
            notes=str(sub.get("notes") or "operator_console_v2"),
            channel_labels=list(acq_cfg.get("channel_labels") or DEFAULT_CHANNEL_LABELS),
        )
        write_session_meta(paths.meta_json, meta)

        protocol_locked = bool(exp.get("protocol_locked", True))
        v2_overrides = exp.get("v2_overrides") if isinstance(exp.get("v2_overrides"), dict) else {}
        v2_path = exp.get("v2_config_path")
        from experiment_game.experiment.v2_config import V2Config

        v2_cfg = V2Config.load_yaml(v2_path) if v2_path else V2Config.load_yaml()
        ignored = v2_cfg.apply_overrides(v2_overrides, protocol_locked=protocol_locked)
        verr = v2_cfg.verify_errors()
        if verr:
            raise RuntimeError("v2 配置无效: " + "; ".join(verr))
        from experiment_game.experiment.session_v2 import (
            diagnose_v2_online_deps,
            probe_v2_weights_missing,
        )

        weights_missing = probe_v2_weights_missing(v2_cfg)
        update_session_meta(
            paths.meta_json,
            v2_config=str(v2_path or "config/v2_session.yaml"),
            v2_config_effective=v2_cfg.to_dict(),
            protocol_locked=protocol_locked,
            v2_overrides_ignored=ignored,
            v2_weights_missing=weights_missing,
            # 采集前尚无 LSL，不能定论 degraded；仅权重缺失时先提示
            v2_degraded=weights_missing,
            seed=exp.get("seed"),
            v2_skips={
                "guidance": bool(exp.get("skip_v2_guidance")),
                "calibration": bool(exp.get("skip_v2_calibration")),
                "gate": bool(exp.get("skip_v2_gate")),
                "game": bool(exp.get("skip_v2_game")),
            },
        )

        self.bridge.broadcast(
            {
                "type": "session_started",
                "session_root": str(paths.root),
                "subject_url": self.subject_url,
                "acq_enabled": acq_on,
                "board_mode": acq_cfg["board_mode"],
                "serial_port": acq_cfg.get("serial_port"),
                "phase_mode": "v2_session",
                # 采集前：仅权重缺失才标演练；LSL 就绪后由 v2_online_status 纠正
                "degraded": weights_missing,
                "degraded_pending_lsl": bool(acq_on) and not weights_missing,
                "save_root": str(save_root),
                "open_subject_page": exp.get("open_subject_page", True),
                "segment": 1,
                "protocol_locked": protocol_locked,
                "timing": {
                    "prep_s": v2_cfg.prep_s,
                    "cue_s": v2_cfg.cue_s,
                    "mi_s": v2_cfg.imagine_s,
                    "iti_s": v2_cfg.iti_s,
                    "fixation_s": v2_cfg.prep_s,
                    "post_mi_hold_s": 0,
                    "rest_s": 0,
                    "transition_s": v2_cfg.iti_s,
                },
                "trial_total_s": v2_cfg.trial_total_s(),
                "v2_config_effective": {
                    "cal_rounds_min": v2_cfg.cal_rounds_min,
                    "cal_rounds_max": v2_cfg.cal_rounds_max,
                    "game_rounds": v2_cfg.game_rounds,
                    "gate_enter_three": v2_cfg.gate_enter_three,
                },
            }
        )

        events = EventLogger(paths.events_jsonl)
        self._events = events
        markers = MarkerPublisher(enabled=acq_on and bool(acq_cfg.get("markers_lsl", True)))
        self._markers = markers
        runner = SessionRunner(events, markers, self.bridge, on_console=print)
        self._runner = runner

        if acq_on:
            try:
                self._start_acquisition_pipeline(
                    paths, meta, acq_cfg, use_synthetic
                )
                # 采集已推 LSL：再确认在线依赖（纠正 session_started 的 pending 状态）
                deps, reasons = diagnose_v2_online_deps(
                    v2_cfg, require_lsl=True, lsl_timeout_s=8.0, on_console=print
                )
                online_ok = deps is not None
                if deps is not None and deps[1] is not None:
                    try:
                        deps[1].close()
                    except Exception:
                        pass
                update_session_meta(
                    paths.meta_json,
                    v2_degraded=not online_ok,
                    v2_online_ready=online_ok,
                    v2_online_fail_reasons=reasons,
                )
                self.bridge.broadcast({
                    "type": "v2_online_status",
                    "degraded": not online_ok,
                    "reason": "ok" if online_ok else "post_acq_check_failed",
                    "message": (
                        "在线判定与微调已启用"
                        if online_ok
                        else ("演练模式：" + ("；".join(reasons) if reasons else "LSL/权重不可用"))
                    ),
                    "reasons": reasons,
                })
                if not online_ok:
                    print(f"[operator] ⚠️ 采集已开但仍无法挂在线推理：{reasons}")
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self._emit_acq_status("error", msg)
                raise RuntimeError(msg) from exc
        else:
            self._emit_acq_status("idle", "本次未开启采集")
            update_session_meta(paths.meta_json, v2_degraded=True, v2_online_ready=False)
            self.bridge.broadcast({
                "type": "v2_online_status",
                "degraded": True,
                "reason": "acq_disabled",
                "message": "演练模式：未开启采集，无 LSL / 无在线微调",
            })

        events.emit(
            "session_start",
            subject_id=subject_id,
            session_id=session_id,
            phase="v2_session",
        )
        markers.push(f"session_start|subject={subject_id}|session={session_id}|phase=v2_session")

        self.bridge.broadcast({"type": "session", "status": "running", "phase": "v2_session"})

        # 诱导页由操作台前端打开；弹窗被拦时前端发 open_subject_page
        timeout = float(exp.get("ready_timeout_s") or 90)
        print(f"[operator] 等待诱导页 ready（{timeout:.0f}s）…")
        try:
            runner.wait_browser_ready(timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"诱导页未在 {timeout:.0f}s 内 ready；请点「重新打开诱导页」并允许弹窗"
            ) from exc

        v2_path = exp.get("v2_config_path")
        summary = runner.run_v2_session(
            config_path=v2_path,
            v2_overrides=v2_overrides,
            protocol_locked=protocol_locked,
            seed=exp.get("seed"),
            skip_guidance=bool(exp.get("skip_v2_guidance")),
            skip_calibration=bool(exp.get("skip_v2_calibration")),
            skip_gate=bool(exp.get("skip_v2_gate")),
            skip_game=bool(exp.get("skip_v2_game")),
        )
        self._runner = None

        events.emit("session_end", subject_id=subject_id, session_id=session_id, phase="v2_session")
        markers.push("session_end|phase=v2_session")

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
            print("[operator] auto_phase4 v2：phase4_v2 + phase4_v2_game…")
            try:
                from experiment_game.offline.phase4_v2 import run as run_p4_cal
                from experiment_game.offline.phase4_v2_game import run as run_p4_game

                run_p4_cal(str(paths.root))
                run_p4_game(str(paths.root))
                from experiment_game.experiment.v2_acceptance import count_phase4_windows

                phase4_result = {
                    "ok": True,
                    "message": "v2 双管道切窗完成",
                    "epochs_dir": str(paths.root / "phase4_v2"),
                    "v2_pipes": count_phase4_windows(paths.root),
                }
            except Exception as exc:  # noqa: BLE001
                phase4_result = {"ok": False, "message": str(exc)}

        from experiment_game.experiment.v2_acceptance import compute_v2_acceptance

        v2_accept = compute_v2_acceptance(
            summary=summary, verify=verify, session_root=paths.root
        )
        update_session_meta(paths.meta_json, v2_summary=summary, v2_acceptance=v2_accept)
        if summary.get("weak_mi"):
            update_session_meta(paths.meta_json, weak_mi=True, gate_status="weak_mi")

        files = self._list_session_files(paths.root)
        self.bridge.broadcast(
            {
                "type": "session_saved",
                "root": str(paths.root),
                "files": files,
                "acq_enabled": acq_on,
                "train_eligible": bool(acq_on and verify.get("passed", False)),
                "verify": verify,
                "phase4": phase4_result,
                "acq_quality": self._acq_quality_for_saved(verify),
                "v2_summary": summary,
                "v2_acceptance": v2_accept,
                "phase_mode": "v2_session",
                "message": "v2 会话已结束" if acq_on else "v2 会话已结束（无 EEG）",
            }
        )
        self.bridge.broadcast({"type": "session", "status": "done"})
        if acq_on:
            self._emit_acq_status("stopped", "录制已停止")
        print(f"[operator] v2 会话目录: {paths.root}")

    def _run_v3_session(self, cfg: Dict[str, Any]) -> None:
        """v3 探针会话：零样本冻结 · A/B 引导 · 拒跑无模型。"""
        sub = cfg["subject"]
        acq_cfg = cfg["acquisition"]
        exp = cfg["experiment"]
        storage = cfg["storage"]
        save_root = Path(storage["save_root"])
        subject_id = sub["subject_id"]
        session_id = sub["session_id"]

        paths = create_session_dir(save_root, subject_id, session_id)
        self._paths = paths
        (paths.root / "run_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._maybe_persist_last_config(cfg)

        use_synthetic = acq_cfg["board_mode"] != "cyton"
        acq_on = bool(acq_cfg["enabled"])
        if not acq_on:
            raise RuntimeError(
                "v3 探针会话必须开启采集（无演练模式）。请勾选采集并确认 LSL 可用。"
            )

        from experiment_game.experiment.v3_config import V3Config
        from experiment_game.experiment.session_v3 import block_order, diagnose_v3_deps

        protocol_locked = bool(exp.get("protocol_locked", True))
        v3_overrides = exp.get("v3_overrides") if isinstance(exp.get("v3_overrides"), dict) else {}
        v3_path = exp.get("v3_config_path")
        v3_cfg = V3Config.load_yaml(v3_path) if v3_path else V3Config.load_yaml()
        ignored = v3_cfg.apply_overrides(v3_overrides, protocol_locked=protocol_locked)
        verr = v3_cfg.verify_errors()
        if verr:
            raise RuntimeError("v3 配置无效: " + "; ".join(verr))

        seed = exp.get("seed")
        b_order = block_order(seed=seed, subject_id=subject_id)

        meta = SessionMeta(
            subject_id=subject_id,
            session_id=session_id,
            phase_mode="v3_session",
            use_synthetic=use_synthetic,
            trial_count=0,
            object="cup",
            scene="home_desk",
            notes=str(sub.get("notes") or "operator_console_v3"),
            channel_labels=list(acq_cfg.get("channel_labels") or DEFAULT_CHANNEL_LABELS),
        )
        write_session_meta(paths.meta_json, meta)
        update_session_meta(
            paths.meta_json,
            v3_config=str(v3_path or "config/v3_session.yaml"),
            v3_config_effective=v3_cfg.to_dict(),
            protocol_locked=protocol_locked,
            v3_overrides_ignored=ignored,
            v3_degraded=False,
            v3_block_order=b_order,
            v3_seed=seed,
        )

        self.bridge.broadcast({
            "type": "session_started",
            "session_root": str(paths.root),
            "subject_url": self.subject_url,
            "acq_enabled": acq_on,
            "board_mode": acq_cfg["board_mode"],
            "serial_port": acq_cfg.get("serial_port"),
            "phase_mode": "v3_session",
            "degraded": False,
            "save_root": str(save_root),
            "open_subject_page": exp.get("open_subject_page", True),
            "segment": 1,
            "protocol_locked": protocol_locked,
            "v3_block_order": b_order,
            "timing": {
                "prep_s": v3_cfg.prep_s,
                "cue_s": v3_cfg.cue_s,
                "mi_s": v3_cfg.imagine_s,
                "iti_s": v3_cfg.iti_s,
                "fixation_s": v3_cfg.prep_s,
                "post_mi_hold_s": 0,
                "rest_s": 0,
                "transition_s": v3_cfg.iti_s,
            },
            "trial_total_s": v3_cfg.trial_total_s(),
            "v3_config_effective": {
                "blocks": v3_cfg.blocks,
                "trials_per_block": v3_cfg.trials_per_block,
                "baseline_rest_s": v3_cfg.baseline_rest_s,
            },
        })

        events = EventLogger(paths.events_jsonl)
        self._events = events
        markers = MarkerPublisher(enabled=acq_on and bool(acq_cfg.get("markers_lsl", True)))
        self._markers = markers
        runner = SessionRunner(events, markers, self.bridge, on_console=print)
        self._runner = runner

        try:
            self._start_acquisition_pipeline(
                paths, meta, acq_cfg, use_synthetic
            )
            deps, reasons = diagnose_v3_deps(v3_cfg, lsl_timeout_s=8.0, on_console=print)
            if deps is None:
                msg = (
                    "v3 自检失败（权重或 LSL 不可用）："
                    + ("；".join(reasons) if reasons else "未知")
                    + "\n请确认 open_operator.bat、采集已开、ckpt 路径正确。"
                )
                raise RuntimeError(msg)
            if deps[1] is not None:
                try:
                    deps[1].close()
                except Exception:
                    pass
            update_session_meta(paths.meta_json, v3_online_ready=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self._emit_acq_status("error", msg)
            raise RuntimeError(msg) from exc

        events.emit("session_start", subject_id=subject_id, session_id=session_id, phase="v3_session")
        markers.push(f"session_start|subject={subject_id}|session={session_id}|phase=v3_session")
        self.bridge.broadcast({"type": "session", "status": "running", "phase": "v3_session"})

        timeout = float(exp.get("ready_timeout_s") or 90)
        print(f"[operator] 等待诱导页 ready（{timeout:.0f}s）…")
        try:
            runner.wait_browser_ready(timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"诱导页未在 {timeout:.0f}s 内 ready；请点「重新打开诱导页」并允许弹窗"
            ) from exc

        summary = runner.run_v3_session(
            config_path=v3_path,
            v3_overrides=v3_overrides,
            protocol_locked=protocol_locked,
            seed=seed,
            subject_id=subject_id,
        )
        self._runner = None

        events.emit("session_end", subject_id=subject_id, session_id=session_id, phase="v3_session")
        markers.push("session_end|phase=v3_session")
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

        update_session_meta(paths.meta_json, v3_summary=summary)
        files = self._list_session_files(paths.root)
        self.bridge.broadcast({
            "type": "session_saved",
            "root": str(paths.root),
            "files": files,
            "acq_enabled": acq_on,
            "train_eligible": False,
            "verify": verify,
            "v3_summary": summary,
            "phase_mode": "v3_session",
            "message": "v3 探针会话已结束",
        })
        self.bridge.broadcast({"type": "session", "status": "done"})
        self._emit_acq_status("stopped", "录制已停止")
        print(f"[operator] v3 会话目录: {paths.root}")

    def _run_v4_session(self, cfg: Dict[str, Any]) -> None:
        """v4 实验前数据质量检测：无模型、连续帽检。"""
        sub = cfg["subject"]
        acq_cfg = cfg["acquisition"]
        exp = cfg["experiment"]
        storage = cfg["storage"]
        save_root = Path(storage["save_root"])
        subject_id = sub["subject_id"]
        session_id = sub["session_id"]

        paths = create_session_dir(save_root, subject_id, session_id)
        self._paths = paths
        (paths.root / "run_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._maybe_persist_last_config(cfg)

        use_synthetic = acq_cfg["board_mode"] != "cyton"
        acq_on = bool(acq_cfg["enabled"])
        if not acq_on:
            raise RuntimeError("v4 质量检测必须开启采集。")

        from experiment_game.experiment.v4_config import V4Config
        from experiment_game.experiment.session_v4 import diagnose_v4_lsl

        v4_overrides = exp.get("v4_overrides") if isinstance(exp.get("v4_overrides"), dict) else {}
        v4_path = exp.get("v4_config_path")
        v4_cfg = V4Config.load_yaml(v4_path) if v4_path else V4Config.load_yaml()
        ignored = v4_cfg.apply_overrides(v4_overrides)
        verr = v4_cfg.verify_errors()
        if verr:
            raise RuntimeError("v4 配置无效: " + "; ".join(verr))

        meta = SessionMeta(
            subject_id=subject_id,
            session_id=session_id,
            phase_mode="v4_session",
            use_synthetic=use_synthetic,
            trial_count=0,
            object="cup",
            scene="home_desk",
            notes=str(sub.get("notes") or "operator_console_v4"),
            channel_labels=list(acq_cfg.get("channel_labels") or DEFAULT_CHANNEL_LABELS),
        )
        write_session_meta(paths.meta_json, meta)
        update_session_meta(
            paths.meta_json,
            v4_config=str(v4_path or "config/v4_session.yaml"),
            v4_config_effective=v4_cfg.to_dict(),
            v4_overrides_ignored=ignored,
        )

        self.bridge.broadcast({
            "type": "session_started",
            "session_root": str(paths.root),
            "subject_url": self.subject_url,
            "acq_enabled": acq_on,
            "board_mode": acq_cfg["board_mode"],
            "serial_port": acq_cfg.get("serial_port"),
            "phase_mode": "v4_session",
            "degraded": False,
            "save_root": str(save_root),
            "open_subject_page": False,
            "v4_config_effective": {
                "duration_s": v4_cfg.duration_s,
                "pass_streak_required": v4_cfg.pass_streak_required,
            },
        })

        events = EventLogger(paths.events_jsonl)
        self._events = events
        markers = MarkerPublisher(enabled=acq_on and bool(acq_cfg.get("markers_lsl", True)))
        self._markers = markers
        runner = SessionRunner(events, markers, self.bridge, on_console=print)
        self._runner = runner

        buf = None
        try:
            self._start_acquisition_pipeline(paths, meta, acq_cfg, use_synthetic)
            buf, reasons = diagnose_v4_lsl(v4_cfg, on_console=print)
            if buf is None:
                msg = "v4 LSL 挂接失败：" + ("；".join(reasons) if reasons else "未知")
                raise RuntimeError(msg)
            update_session_meta(paths.meta_json, v4_lsl_ready=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self._emit_acq_status("error", msg)
            raise RuntimeError(msg) from exc

        events.emit("session_start", subject_id=subject_id, session_id=session_id, phase="v4_session")
        markers.push(f"session_start|subject={subject_id}|session={session_id}|phase=v4_session")
        self.bridge.broadcast({"type": "session", "status": "running", "phase": "v4_session"})

        try:
            summary = runner.run_v4_session(
                buf,
                config_path=v4_path,
                v4_overrides=v4_overrides,
                session_dir=paths.root,
            )
        finally:
            if buf is not None:
                try:
                    buf.close()
                except Exception:
                    pass
        self._runner = None

        events.emit("session_end", subject_id=subject_id, session_id=session_id, phase="v4_session")
        markers.push("session_end|phase=v4_session")
        self._shutdown_session_resources()

        update_session_meta(paths.meta_json, v4_summary=summary)
        files = self._list_session_files(paths.root)
        self.bridge.broadcast({
            "type": "session_saved",
            "root": str(paths.root),
            "files": files,
            "acq_enabled": acq_on,
            "train_eligible": False,
            "v4_summary": summary,
            "phase_mode": "v4_session",
            "message": "v4 质量检测已结束",
        })
        self.bridge.broadcast({"type": "session", "status": "done"})
        self._emit_acq_status("stopped", "录制已停止")
        print(f"[operator] v4 会话目录: {paths.root}")

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
                # v2 会话：走双管道切窗
                meta_path = Path(target) / "session.meta.json"
                is_v2 = False
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        is_v2 = meta.get("phase_mode") == "v2_session"
                    except Exception:
                        pass
                if is_v2:
                    from experiment_game.offline.phase4_v2 import run as run_p4_cal
                    from experiment_game.offline.phase4_v2_game import run as run_p4_game
                    from experiment_game.experiment.v2_acceptance import count_phase4_windows

                    run_p4_cal(str(target))
                    run_p4_game(str(target))
                    pipes = count_phase4_windows(Path(target))
                    self.bridge.broadcast({
                        "type": "phase4_ack",
                        "ok": True,
                        "message": "v2 双管道切窗完成",
                        "epochs_dir": str(Path(target) / "phase4_v2"),
                        "path": str(target),
                        "v2_pipes": pipes,
                        "summary": {"n": pipes.get("pipes", {}).get("phase4_v2", {}).get("n_windows")},
                    })
                    return
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

    def _acq_quality_for_saved(self, verify: Dict[str, Any]) -> Dict[str, Any]:
        q = verify.get("quality")
        if isinstance(q, dict) and q:
            return q
        if self._last_acq_quality:
            return dict(self._last_acq_quality)
        return {}

    def _shutdown_session_resources(self) -> None:
        self._stop_link_monitor()
        acq = self._acq
        self._acq = None
        if acq is not None:
            try:
                report = acq.stop()
                print(f"[operator] 录制停止: {report.get('message')}")
                quality = report.get("quality")
                if isinstance(quality, dict) and quality:
                    self._last_acq_quality = quality
                    q_msg = quality.get("drop_rate_pct")
                    if q_msg is not None:
                        print(
                            f"[operator] 录制质量: drop_rate_pct={q_msg} "
                            f"severity={quality.get('severity', '?')}"
                        )
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

    def _build_acquisition_facade(
        self,
        acq_cfg: Dict[str, Any],
        channel_labels: List[str],
        use_synthetic: bool,
    ) -> AcquisitionFacade:
        filt = acq_cfg.get("filter") or {}
        return AcquisitionFacade(
            use_synthetic=use_synthetic,
            serial_port=str(acq_cfg.get("serial_port") or "COM5"),
            channel_labels=channel_labels,
            filter_enabled=bool(filt.get("enabled", True)),
            bandpass_low_hz=float(filt.get("bandpass_low_hz", 0.5)),
            bandpass_high_hz=float(filt.get("bandpass_high_hz", 45.0)),
            notch_low_hz=float(filt.get("notch_low_hz", 49.0)),
            notch_high_hz=float(filt.get("notch_high_hz", 51.0)),
        )

    def _on_acq_link_event(self, event: Dict[str, Any]) -> None:
        kind = str(event.get("kind") or "")
        message = str(event.get("message") or kind)
        print(f"[operator] [链路] {message}")
        if self._markers is None:
            return
        if kind not in (
            "stall",
            "reconnect_attempt",
            "reconnect_ok",
            "reconnect_fail",
            "link_dead",
        ):
            return
        gap_s = event.get("gap_s", "")
        attempt = event.get("attempt", "")
        ok = 1 if kind == "reconnect_ok" else 0
        self._markers.push(
            f"acq_reconnect|kind={kind}|attempt={attempt}|gap_s={gap_s}|ok={ok}"
        )

    def _start_acquisition_pipeline(
        self,
        paths: SessionPaths,
        meta: SessionMeta,
        acq_cfg: Dict[str, Any],
        use_synthetic: bool,
    ) -> None:
        """F1 预检 + 启采集 + health_check + F3 链路监控。"""
        self._emit_acq_status("connecting", "正在检查无线链路…")
        self._acq = self._build_acquisition_facade(
            acq_cfg, meta.channel_labels, use_synthetic
        )

        probe = self._acq.preflight_probe()
        if not probe.get("ok"):
            guidance = str(probe.get("guidance") or "链路预检失败")
            self._emit_acq_status("error", guidance)
            raise RuntimeError(guidance)

        fw = str(probe.get("firmware_line") or "")
        if fw:
            update_session_meta(
                paths.meta_json,
                cyton_firmware=fw,
                cyton_serial_port=self._acq.serial_port,
            )
        self._link_firmware = fw

        self._emit_acq_status("connecting", "正在启动采集…")
        self._acq.create(on_link_event=self._on_acq_link_event)
        self._acq.start(paths.eeg_csv)
        self._acq.health_check()
        self._emit_acq_status("recording", "录制中")
        self._start_link_monitor()

    def _start_link_monitor(self) -> None:
        self._stop_link_monitor()
        self._link_monitor_stop.clear()
        self._link_prev_pushed = 0
        if self._acq is not None:
            try:
                st = self._acq.manager.get_status()
                self._link_prev_pushed = int(st.get("samples_pushed") or 0)
            except Exception:  # noqa: BLE001
                pass
        self._link_monitor_thread = threading.Thread(
            target=self._link_monitor_loop,
            name="LinkMonitor",
            daemon=True,
        )
        self._link_monitor_thread.start()

    def _stop_link_monitor(self) -> None:
        self._link_monitor_stop.set()
        t = self._link_monitor_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._link_monitor_thread = None

    def _short_firmware(self, raw: str) -> str:
        try:
            from lsl_connect.cyton_link import short_firmware_display

            return short_firmware_display(raw)
        except ImportError:
            text = re.sub(r"[^\x20-\x7E]", "", str(raw or "")).strip()
            if not text:
                return ""
            if "Firmware:" in text:
                return text.split("Firmware:", 1)[-1].strip()[:32]
            match = re.search(r"OpenBCI[\w .-]{0,40}", text, re.IGNORECASE)
            return match.group(0) if match else text[:48]

    def _link_monitor_loop(self) -> None:
        interval = 2.0
        warned_dead = False
        while not self._link_monitor_stop.wait(interval):
            acq = self._acq
            if acq is None:
                continue
            try:
                st = acq.manager.get_status()
            except Exception:  # noqa: BLE001
                continue

            pushed = int(st.get("samples_pushed") or 0)
            hz = (pushed - self._link_prev_pushed) / interval
            self._link_prev_pushed = pushed

            link_stats = st.get("link_stats") or {}
            last_connect = st.get("last_connect") or {}
            firmware = self._link_firmware or str(last_connect.get("firmware") or "")
            fw_short = self._short_firmware(firmware)

            last_event = link_stats.get("last_event") or {}
            link_dead = bool(link_stats.get("link_dead"))
            guidance = ""
            if link_dead:
                guidance = str(
                    last_event.get("message")
                    or "无线断流，自动重连失败：请检查 Cyton 电量、dongle 距离后重开机"
                ).split("\n")[0]

            self.bridge.broadcast(
                {
                    "type": "link_status",
                    "port": st.get("serial_port_raw") or st.get("serial_port"),
                    "firmware": fw_short,
                    "state": st.get("state"),
                    "worker_running": st.get("worker_running"),
                    "streaming_hz": round(hz, 1),
                    "samples_pushed": pushed,
                    "gap_samples": int(st.get("gap_samples") or 0),
                    "reconnect_ok": int(link_stats.get("reconnect_ok") or 0),
                    "reconnect_fail": int(link_stats.get("reconnect_fail") or 0),
                    "link_dead": link_dead,
                    "last_event": last_event,
                    "guidance": guidance,
                }
            )

            if link_dead and not warned_dead:
                warned_dead = True
                self._emit_acq_status("error", guidance)
                self.bridge.broadcast(
                    {
                        "type": "v3_warn",
                        "code": "link_dead",
                        "message": guidance,
                    }
                )

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
