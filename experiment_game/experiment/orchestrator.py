"""操作台编排：空闲起服务 → 校验配置 → 开会话 → 等待诱导页 ready → SessionRunner。"""

from __future__ import annotations

import json
import os
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
from experiment_game.experiment.timing import DEFAULT_TIMING
from experiment_game.experiment.local_ports import check_ports_free, format_port_conflict
from experiment_game.experiment.ws_bridge import WsBridge
from experiment_game.offline.phase4_service import run_phase4_for_session

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEB_ROOT = _PKG_ROOT / "web"


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
            self._handle_run_phase4(str(msg.get("path") or ""))
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
        for name in ("ready", "continue", "abort", "gate_ok"):
            self.bridge.clear_event(name)
        self.bridge.paused = False
        self.bridge.reject_requested = False

        sub = cfg["subject"]
        acq_cfg = cfg["acquisition"]
        exp = cfg["experiment"]
        storage = cfg["storage"]
        save_root = Path(storage["save_root"])

        paths = create_session_dir(save_root, sub["subject_id"], sub["session_id"])
        self._paths = paths

        # 会话快照（便于复现；UI-2 正式化）
        (paths.root / "run_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._maybe_persist_last_config(cfg)

        use_synthetic = acq_cfg["board_mode"] != "cyton"
        acq_on = bool(acq_cfg["enabled"])
        meta = SessionMeta(
            subject_id=sub["subject_id"],
            session_id=sub["session_id"],
            phase_mode="phase2_full",
            use_synthetic=use_synthetic if acq_on else True,
            trial_count=int(exp["acquire_trials"]),
            object="cup",
            scene="home_desk",
            notes=str(sub.get("notes") or "operator_console"),
            channel_labels=list(
                acq_cfg.get("channel_labels") or DEFAULT_CHANNEL_LABELS
            ),
        )
        write_session_meta(paths.meta_json, meta)

        self.bridge.broadcast(
            {
                "type": "session_started",
                "session_root": str(paths.root),
                "subject_url": self.subject_url,
                "acq_enabled": acq_on,
                "board_mode": acq_cfg["board_mode"],
                "serial_port": acq_cfg.get("serial_port"),
                "acquire_trials": exp["acquire_trials"],
                "save_root": str(save_root),
                "open_subject_page": exp.get("open_subject_page", True),
            }
        )

        events = EventLogger(paths.events_jsonl)
        self._events = events
        markers = MarkerPublisher(enabled=acq_on and bool(acq_cfg.get("markers_lsl", True)))
        self._markers = markers

        phase_cfg = Phase2Config(
            acquire_trials=int(exp["acquire_trials"]),
            learn_trials_per_step=int(exp["learn_trials_per_step"]),
            seed=exp.get("seed"),
            skip_adapt=bool(exp.get("skip_adapt", False)),
            skip_learn=bool(exp.get("skip_learn", False)),
            skip_gate=bool(exp.get("skip_gate", False)),
            auto_continue=False,
            rotate_objects=True,
            rotate_scenes=True,
        )
        runner = SessionRunner(events, markers, self.bridge, timing=DEFAULT_TIMING, config=phase_cfg)

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
            subject_id=sub["subject_id"],
            session_id=sub["session_id"],
            phase="phase2",
        )
        markers.push(
            f"session_start|subject={sub['subject_id']}|session={sub['session_id']}|phase=phase2"
        )

        self.bridge.broadcast({"type": "session", "status": "running", "phase": "waiting_ready"})

        if exp.get("open_subject_page", True):
            # 后端再开一次，作为弹窗拦截时的兜底（与前端 window.open 并存）
            try:
                webbrowser.open(self.subject_url)
            except Exception as exc:  # noqa: BLE001
                print(f"[operator] 打开诱导页失败: {exc}", file=sys.stderr)

        timeout = float(exp.get("ready_timeout_s") or 60)
        print(f"[operator] 等待诱导页 ready（{timeout:.0f}s）…")
        try:
            runner.wait_browser_ready(timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"诱导页未在 {timeout:.0f}s 内 ready；请点「重新打开诱导页」并允许弹窗"
            ) from exc

        runner.run_all()

        events.emit(
            "session_end",
            subject_id=sub["subject_id"],
            session_id=sub["session_id"],
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
            print("[operator] auto_phase4：仅 acquire + 未 reject …")
            phase4_result = run_phase4_for_session(
                paths.root,
                repo_root=self.repo_root,
            )
            print(
                f"[operator] Phase4: ok={phase4_result.get('ok')} "
                f"{phase4_result.get('message')} → {phase4_result.get('epochs_dir')}"
            )

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
                "message": "会话已结束" if acq_on else "会话已结束（无 EEG，不可训练切窗）",
            }
        )
        self.bridge.broadcast({"type": "session", "status": "done"})
        if acq_on:
            self._emit_acq_status("stopped", "录制已停止")
        print(f"[operator] 会话目录: {paths.root}")

    def _handle_run_phase4(self, path: str) -> None:
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

        def _job() -> None:
            try:
                result = run_phase4_for_session(Path(target), repo_root=self.repo_root)
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
