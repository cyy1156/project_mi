"""操作台编排：空闲起服务 → 校验配置 → 开会话 → 等待诱导页 ready → SessionRunner。"""

from __future__ import annotations

import json
import os
import atexit
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# adapt_engine / preprocess_lab 在 code/ 下；操作台入口须先于任何校验导入
_REPO_BOOT = Path(__file__).resolve().parents[2]
for _boot in (_REPO_BOOT, _REPO_BOOT / "code", _REPO_BOOT / "code" / "preprocess_lab"):
    _bs = str(_boot)
    if _bs not in sys.path:
        sys.path.insert(0, _bs)

from experiment_game.core.atomic_io import atomic_write_json
from experiment_game.experiment.alignment import write_alignment_bundle
from experiment_game.experiment.defaults_store import (
    defaults_path,
    load_operator_defaults,
    save_operator_defaults,
)
from experiment_game.experiment.session_finalize import ensure_crash_artifacts
from experiment_game.experiment.session_layout import finalize_session_layout
from experiment_game.experiment.trial_scoring import session_score_max_openbmi
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

# 2026-08-29 重构：offline/tools 实现改为依赖注入（见构造函数 phase4_runner/ft_runner），
# 本模块不再顶层 import offline/tools；惰性回退仅存在于 *_resolve_* 接缝方法内。

_PKG_ROOT = Path(__file__).resolve().parents[1]


def _subject_feedback_mode(cfg: Dict[str, Any]) -> str:
    ui = cfg.get("ui")
    if isinstance(ui, dict):
        mode = str(ui.get("subject_feedback_mode") or "none").strip()
        if mode in ("none", "arm_reach"):
            return mode
    return "none"
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
        phase4_runner: Optional[Callable[..., Dict[str, Any]]] = None,
        phase4_v2_pair_runner: Optional[Callable[..., Any]] = None,
        ft_runner: Optional[Callable[..., Dict[str, Any]]] = None,
        ws_token: Optional[str] = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else _REPO_ROOT
        self.web_root = Path(web_root) if web_root else _WEB_ROOT
        self.http_port = http_port
        self.ws_port = ws_port
        self.serve_host = serve_host
        # 依赖注入接缝（2026-08-29 重构）：入口层（tools/）负责注入具体实现；
        # 未注入时 *_resolve_* 方法回退到惰性导入（过渡期，行为不变）。
        self._phase4_runner = phase4_runner
        self._phase4_v2_pair_runner = phase4_v2_pair_runner
        self._ft_runner = ft_runner
        # 控制面 token：未传则自动生成；操作台 URL 带 ?token=，诱导页 continue 无需 token
        self.ws_token = (str(ws_token).strip() if ws_token else "") or WsBridge.make_token()
        lan = serve_host not in ("127.0.0.1", "localhost", "::1")
        origins = (
            None
            if lan
            else [
                # None 匹配"无 Origin 头"的连接：本机脚本/冒烟测试不带 Origin，
                # 缺了它会被 websockets 17 直接 403（浏览器仍受白名单约束）
                None,
                f"http://127.0.0.1:{http_port}",
                f"http://localhost:{http_port}",
            ]
        )
        self.bridge = WsBridge(
            host=serve_host,
            port=ws_port,
            auth_token=self.ws_token,
            allowed_origins=origins,
        )
        self.http = StaticServer(self.web_root, host=serve_host, port=http_port)
        self._lock = threading.Lock()
        self._busy = False
        self._worker: Optional[threading.Thread] = None
        self._acq: Optional[AcquisitionFacade] = None
        self._last_acq_quality: Dict[str, Any] = {}
        self._live_capture = None  # experiment.live_capture.LiveEegCapture：Bus CSV 单写
        self._v2_injected_deps = None
        self._v3_injected_deps = None
        self._events: Optional[EventLogger] = None
        self._markers: Optional[MarkerPublisher] = None
        self._paths: Optional[SessionPaths] = None
        self._layout_finalized = False
        self._crash_atexit_registered = False
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
        # 被试登录 + 离线微调
        self._active_subject: Optional[str] = None
        self._active_subject_info: Optional[Dict[str, Any]] = None
        self._active_sim_mode: bool = False
        self._ft_busy = False
        self._ft_worker: Optional[threading.Thread] = None
        self._eval_busy = False

    # ---- 依赖注入接缝（2026-08-29 重构，见 docs/重构实施方案_20260829.md §3.3）----
    # 入口层 tools/ 注入具体实现；未注入时回退惰性导入（过渡期，行为不变）。
    def _resolve_phase4_runner(self) -> Callable[..., Dict[str, Any]]:
        if self._phase4_runner is not None:
            return self._phase4_runner
        from experiment_game.offline.phase4_service import run_phase4_for_session

        return run_phase4_for_session

    def _resolve_phase4_v2_pair(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        if self._phase4_v2_pair_runner is not None:
            return self._phase4_v2_pair_runner
        from experiment_game.offline.phase4_v2 import run as run_p4_cal
        from experiment_game.offline.phase4_v2_game import run as run_p4_game

        return run_p4_cal, run_p4_game

    def _resolve_ft_runner(self) -> Callable[..., Dict[str, Any]]:
        if self._ft_runner is not None:
            return self._ft_runner
        from experiment_game.pipeline.finetune import run_subject_finetune

        return run_subject_finetune

    @property
    def operator_url(self) -> str:
        # 本机浏览器仍用 127.0.0.1；LAN 地址在 start() 额外打印
        tok = f"?token={self.ws_token}" if self.ws_token else ""
        return f"http://127.0.0.1:{self.http_port}/operator.html{tok}#setup"

    @property
    def subject_url(self) -> str:
        # 不强制 ?ws=127.0.0.1：页面按 location.hostname 自动连 WS，
        # 本机与局域网监控端都能连上（需 open_operator_lan / --host 0.0.0.0）。
        # 诱导页无需 token（continue/ready 为公开事件）。
        return f"http://127.0.0.1:{self.http_port}/"

    def _model_presets_payload(self) -> Dict[str, Any]:
        from experiment_game.experiment.model_presets import (
            active_weights_from_yaml,
            list_model_presets,
        )

        return {
            "model_presets": list_model_presets(
                subject_id=self._active_subject,
                sim_mode=self._active_sim_mode,
            ),
            "active_weights": active_weights_from_yaml(),
        }

    @staticmethod
    def _weights_from_cfg(cfg_obj: Any) -> Dict[str, Any]:
        from experiment_game.experiment.model_presets import (
            match_preset_id,
            resolve_model_display_label,
            resolve_weight_display_label,
            short_weight_label,
        )

        task = getattr(cfg_obj, "s3_task_ckpt", "") or ""
        three = getattr(cfg_obj, "s3_three_ckpt", "") or ""
        readout = getattr(cfg_obj, "readout_mode", "") or ""
        preset_id = match_preset_id(task, three)
        if readout.lower() == "e1f":
            preset_id = "e1f_four_member"
        weight_label = resolve_weight_display_label(
            task=task,
            three=three,
            preset_id=preset_id,
            readout_mode=readout,
        )
        model_label = resolve_model_display_label(
            task=task,
            three=three,
            preset_id=preset_id,
            readout_mode=readout,
        )
        return {
            "task": task,
            "three": three,
            "preset_id": preset_id,
            "readout_mode": readout,
            "label": weight_label,
            "weight_label": weight_label,
            "model_label": model_label,
            "short_label": short_weight_label(three or task),
        }

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
        if self.ws_token:
            print(f"WS 控制 token 已启用（abort/gate/operator 需带 token；诱导页 continue 免 token）")
        if self.serve_host not in ("127.0.0.1", "localhost"):
            print(f"绑定地址: {self.serve_host}:{self.http_port} / WS :{self.ws_port}")
            lan = _lan_ipv4_addrs()
            if lan:
                for ip in lan:
                    tok = f"?token={self.ws_token}" if self.ws_token else ""
                    print(f"监控端打开: http://{ip}:{self.http_port}/operator.html{tok}#setup")
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
        """WS 入站路由（W6：dispatch table 在 ws_dispatch 模块）。"""
        mtype = msg.get("type")
        if not mtype or not isinstance(mtype, str):
            return

        # 特殊：operator + split_session
        if mtype == "operator" and str(msg.get("action") or "") == "split_session":
            self._handle_split_request()
            return

        from experiment_game.experiment.ws_dispatch import build_ws_dispatch_table

        handler = build_ws_dispatch_table(self).get(mtype)
        if handler is None:
            return
        handler(msg)

    def _ws_dispatch_table(self) -> Dict[str, Callable[[Dict[str, Any]], None]]:
        from experiment_game.experiment.ws_dispatch import build_ws_dispatch_table

        return build_ws_dispatch_table(self)

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

    def _handle_subject_login(self, msg: Dict[str, Any]) -> None:
        sim_mode = bool(msg.get("sim_mode"))
        try:
            if sim_mode:
                from experiment_game.experiment.sim.sim_registry import login_sim_subject

                sid = str(msg.get("subject_id") or "").strip().upper()
                info = login_sim_subject(sid, repo_root=self.repo_root)
                info["sim_mode"] = True
            else:
                from experiment_game.experiment.subject_registry import login_subject

                sid = str(msg.get("subject_id") or "").strip()
                info = login_subject(
                    sid,
                    display_name=str(msg.get("display_name") or ""),
                    notes=str(msg.get("notes") or ""),
                    repo_root=self.repo_root,
                )
                info["sim_mode"] = False
            self._active_subject = info["subject_id"]
            self._active_subject_info = info
            self._active_sim_mode = sim_mode
            self.bridge.broadcast({
                "type": "subject_login_ack",
                "ok": True,
                **info,
                **self._model_presets_payload(),
            })
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {"type": "subject_login_ack", "ok": False, "message": str(exc)}
            )

    def _handle_subject_logout(self) -> None:
        self._active_subject = None
        self._active_subject_info = None
        self._active_sim_mode = False
        self.bridge.broadcast({"type": "subject_logout_ack", "ok": True})

    def _handle_subject_info(self, msg: Dict[str, Any]) -> None:
        sid = str(msg.get("subject_id") or self._active_subject or "").strip()
        try:
            if self._active_sim_mode:
                from experiment_game.experiment.sim.sim_registry import (
                    build_sim_index,
                    current_sim_model_paths,
                    list_campaigns,
                    list_sim_sessions,
                    validate_sim_subject_id,
                )

                sid = validate_sim_subject_id(sid)
                info = {
                    "subject_id": sid,
                    "sim_mode": True,
                    "sessions": list_sim_sessions(sid, repo_root=self.repo_root),
                    "current_weights": current_sim_model_paths(sid, repo_root=self.repo_root),
                    "index": build_sim_index(sid, repo_root=self.repo_root),
                    "campaigns": list_campaigns(sid, repo_root=self.repo_root),
                }
            else:
                from experiment_game.experiment.subject_registry import (
                    build_index,
                    current_model_paths,
                    list_sessions,
                    validate_subject_id,
                )

                sid = validate_subject_id(sid)
                info = {
                    "subject_id": sid,
                    "sim_mode": False,
                    "sessions": list_sessions(sid, repo_root=self.repo_root),
                    "current_weights": current_model_paths(sid, repo_root=self.repo_root),
                    "index": build_index(sid, repo_root=self.repo_root),
                }
            self.bridge.broadcast({"type": "subject_info_ack", "ok": True, **info})
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {"type": "subject_info_ack", "ok": False, "message": str(exc)}
            )

    def _handle_finetune_start(self, msg: Dict[str, Any]) -> None:
        if self._ft_busy:
            self.bridge.broadcast(
                {
                    "type": "finetune_ack",
                    "ok": False,
                    "message": "已有微调任务在运行",
                }
            )
            return
        if self._busy:
            self.bridge.broadcast(
                {
                    "type": "finetune_ack",
                    "ok": False,
                    "message": "采集会话进行中，请结束后再微调",
                }
            )
            return

        sid = str(msg.get("subject_id") or self._active_subject or "").strip()
        paths_raw = msg.get("session_paths") or msg.get("sessions") or []
        exclude_invalid = bool(msg.get("exclude_invalid"))
        use_replay = msg.get("use_replay")
        no_replay = bool(msg.get("no_replay"))
        if use_replay is not None:
            no_replay = not bool(use_replay)
        replay_ratio = float(msg.get("replay_ratio", 0.10))
        if not (replay_ratio == replay_ratio):  # NaN
            replay_ratio = 0.10
        early_stop = bool(msg.get("early_stop", True))
        if msg.get("no_early_stop"):
            early_stop = False

        def _safe_msg_int(key: str, default: int) -> int:
            v = msg.get(key, default)
            try:
                if v is None or (isinstance(v, float) and v != v):
                    return default
                return int(float(v))
            except (TypeError, ValueError):
                return default

        max_epochs = _safe_msg_int("max_epochs", 20)
        patience = _safe_msg_int("patience", 5)
        deterministic = bool(msg.get("deterministic", True))
        if msg.get("no_deterministic"):
            deterministic = False
        seed_i = _safe_msg_int("seed", 42)
        ft_epochs = _safe_msg_int("epochs", 5)
        leave_next_mode = bool(msg.get("leave_next_mode"))
        eval_run_id = str(msg.get("eval_run_id") or "").strip().lower()
        campaign_manifest_path = msg.get("campaign_manifest")
        use_ramp_replay = bool(msg.get("use_ramp_replay_defaults", leave_next_mode))

        def _job() -> None:
            self._ft_busy = True
            try:
                run_subject_finetune = self._resolve_ft_runner()

                if self._active_sim_mode:
                    from experiment_game.experiment.sim.sim_registry import (
                        new_sim_ft_run_dir,
                        validate_sim_subject_id,
                    )

                    subject_id = validate_sim_subject_id(sid)
                    out_dir = new_sim_ft_run_dir(subject_id, repo_root=self.repo_root)
                else:
                    from experiment_game.experiment.subject_registry import (
                        new_ft_run_dir,
                        validate_subject_id,
                    )

                    subject_id = validate_subject_id(sid)
                    out_dir = new_ft_run_dir(subject_id, repo_root=self.repo_root)
                session_dirs = [Path(p) for p in paths_raw]
                if not self._active_sim_mode:
                    # 需求 2026-08-30 二.2：v3 微调仅使用 v3 模块自采数据，屏蔽其他模块历史数据
                    from experiment_game.experiment.subject_registry import (
                        _read_session_phase_mode,
                    )

                    _v3_dirs: List[Path] = []
                    _excluded: List[Path] = []
                    for _d in session_dirs:
                        pm = _read_session_phase_mode(_d)
                        if pm == "v3_session":
                            _v3_dirs.append(_d)
                        else:
                            _excluded.append(_d)
                    if _excluded:
                        self.bridge.broadcast({
                            "type": "finetune_progress",
                            "level": "warning",
                            "message": (
                                f"已屏蔽 {len(_excluded)} 个非 v3 会话"
                                "（v3 微调仅使用 v3 自采数据）："
                                + ", ".join(d.name for d in _excluded[:6])
                            ),
                        })
                    session_dirs = _v3_dirs
                heldout_dirs: List[Path] = []
                job_no_replay = no_replay
                job_replay_ratio = replay_ratio
                ramp_stage_i: Optional[int] = None
                if leave_next_mode and campaign_manifest_path and eval_run_id:
                    from experiment_game.experiment.sim.campaign import load_campaign
                    from experiment_game.experiment.sim.ramp import (
                        completed_by_run,
                        ft_replay_recommendation,
                        leave_next_train_runs,
                        ramp_stage,
                    )

                    manifest = load_campaign(campaign_manifest_path)
                    train_pairs = leave_next_train_runs(manifest, eval_run_id)
                    if not train_pairs:
                        raise ValueError(
                            f"Leave-Next：eval {eval_run_id} 之前无已完成 session 可训练"
                        )
                    session_dirs = [Path(p) for _, p in train_pairs]
                    done = completed_by_run(manifest)
                    eval_path = done.get(str(eval_run_id).strip().lower())
                    if not eval_path:
                        raise ValueError(
                            f"Leave-Next：找不到 eval session 目录（{eval_run_id}）"
                        )
                    heldout_dirs = [Path(eval_path)]
                    ramp_stage_i = ramp_stage(manifest, eval_run_id)
                    if use_ramp_replay:
                        rec = ft_replay_recommendation(ramp_stage_i)
                        job_no_replay = not bool(rec.get("use_replay"))
                        job_replay_ratio = float(rec.get("replay_ratio") or 0.0)
                if not session_dirs:
                    raise ValueError("未选择 session")
                self.bridge.broadcast(
                    {
                        "type": "finetune_progress",
                        "stage": "start",
                        "out_dir": str(out_dir),
                    }
                )
                from experiment_game.pipeline.ft_policy import load_ft_policy

                ft_policy = load_ft_policy()
                ft_scope = str(ft_policy.get("ft_scope") or "all4")
                if self._active_sim_mode:
                    from experiment_game.experiment.sim.sim_registry import (
                        sim_models_current,
                    )

                    subject_models_dir = sim_models_current(
                        subject_id, repo_root=self.repo_root
                    ).parent
                else:
                    from experiment_game.experiment.subject_registry import (
                        models_current_dir,
                    )

                    subject_models_dir = models_current_dir(
                        subject_id, repo_root=self.repo_root
                    ).parent

                if ft_scope == "all4":
                    from experiment_game.pipeline.e1f_all4_ft import run_e1f_all4_finetune

                    result = run_e1f_all4_finetune(
                        session_dirs,
                        out_dir,
                        subject_models_dir=subject_models_dir,
                        exclude_invalid=exclude_invalid,
                        no_replay=job_no_replay,
                        replay_ratio=job_replay_ratio,
                        epochs=ft_epochs,
                        early_stop=early_stop,
                        max_epochs=max_epochs,
                        patience=patience,
                        deterministic=deterministic,
                        seed=seed_i,
                        verbose=False,
                        heldout_session_dirs=heldout_dirs or None,
                    )
                else:
                    result = run_subject_finetune(
                        session_dirs,
                        out_dir,
                        exclude_invalid=exclude_invalid,
                        no_replay=job_no_replay,
                        replay_ratio=job_replay_ratio,
                        epochs=ft_epochs,
                        early_stop=early_stop,
                        max_epochs=max_epochs,
                        patience=patience,
                        deterministic=deterministic,
                        seed=seed_i,
                        verbose=False,
                        heldout_session_dirs=heldout_dirs or None,
                    )
                    result = dict(result)
                    result.setdefault("ft_scope", ft_scope)

                release_pass = bool(result.get("release_pass"))
                auto_promote = bool(ft_policy.get("auto_promote_after_ft", True))
                force_on_fail = bool(ft_policy.get("force_promote_on_gate_fail", True))
                should_promote = auto_promote and (release_pass or force_on_fail)
                force_promoted = False
                promote_info = None
                if should_promote:
                    from experiment_game.experiment.ft_promote_extras import (
                        write_force_promote_warning,
                    )

                    reason = (
                        "auto_promote_pass"
                        if release_pass
                        else "auto_force_promote_on_gate_fail"
                    )
                    if not release_pass and force_on_fail:
                        write_force_promote_warning(
                            out_dir,
                            release_gate=result.get("release_gate") or {},
                            ft_scope=str(result.get("ft_scope") or ft_scope),
                            subject_id=str(subject_id),
                            reason=reason,
                        )
                        force_promoted = True
                    if self._active_sim_mode:
                        from experiment_game.experiment.sim.sim_registry import (
                            promote_sim_ft_to_current,
                        )

                        promote_info = promote_sim_ft_to_current(
                            subject_id,
                            out_dir,
                            repo_root=self.repo_root,
                            reason=reason,
                        )
                    else:
                        from experiment_game.experiment.subject_registry import (
                            promote_ft_to_current,
                        )

                        promote_info = promote_ft_to_current(
                            subject_id,
                            out_dir,
                            repo_root=self.repo_root,
                            reason=reason,
                        )

                task_ckpt = out_dir / "best_task.pt"
                three_ckpt = out_dir / "best_three.pt"
                presets_payload = self._model_presets_payload()
                # 刚训完的本轮必须出现在预设里（避免扫描时机/路径遗漏）
                if task_ckpt.is_file() and three_ckpt.is_file():
                    from experiment_game.experiment.subject_registry import rel_repo_path

                    task_rel = rel_repo_path(task_ckpt, repo_root=self.repo_root)
                    three_rel = rel_repo_path(three_ckpt, repo_root=self.repo_root)
                    stamp = out_dir.name
                    subj_key = str(subject_id or "")
                    presets = list(presets_payload.get("model_presets") or [])
                    if not any(
                        str(p.get("task") or "").replace("\\", "/").endswith(
                            f"/ft_runs/{stamp}/best_task.pt"
                        )
                        for p in presets
                    ):
                        presets.insert(
                            2,
                            {
                                "id": f"ft_{stamp}",
                                "label": f"{subj_key} · FT {stamp}",
                                "task": task_rel,
                                "three": three_rel,
                                "ok": True,
                                "subject_id": subj_key,
                                "kind": "ft_run",
                                "ft_stamp": stamp,
                                "release_pass": bool(result.get("release_pass")),
                            },
                        )
                        presets_payload["model_presets"] = presets
                self.bridge.broadcast(
                    {
                        "type": "finetune_done",
                        "ok": True,
                        "subject_id": subject_id,
                        "out_dir": result["out_dir"],
                        "ft_task_ckpt": str(task_ckpt) if task_ckpt.is_file() else "",
                        "ft_three_ckpt": str(three_ckpt) if three_ckpt.is_file() else "",
                        "weights_saved": task_ckpt.is_file() and three_ckpt.is_file(),
                        "release_gate": result["release_gate"],
                        "release_pass": result["release_pass"],
                        "three_heldout": result["three"]["acc_after_heldout"],
                        "task_heldout": result["task"]["acc_after_heldout"],
                        "three_ft": result["three"].get("ft"),
                        "report_path": str(out_dir / "report.md"),
                        "use_replay": not job_no_replay,
                        "replay_ratio": 0.0 if job_no_replay else job_replay_ratio,
                        "early_stop": early_stop,
                        "max_epochs": max_epochs,
                        "patience": patience,
                        "deterministic": deterministic,
                        "seed": seed_i,
                        "fixed_epochs": ft_epochs,
                        "leave_next_mode": leave_next_mode,
                        "eval_run_id": eval_run_id or None,
                        "ramp_stage": ramp_stage_i,
                        "train_session_dirs": [str(p) for p in session_dirs],
                        "ft_scope": str(result.get("ft_scope") or ft_scope),
                        "auto_promoted": bool(should_promote),
                        "force_promoted": bool(force_promoted),
                        "promote": promote_info,
                        **presets_payload,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self.bridge.broadcast(
                    {"type": "finetune_done", "ok": False, "message": str(exc)}
                )
            finally:
                self._ft_busy = False

        self._ft_worker = threading.Thread(target=_job, name="finetune-job", daemon=True)
        self._ft_worker.start()
        self.bridge.broadcast({"type": "finetune_ack", "ok": True, "message": "微调已开始"})

    def _handle_finetune_promote(self, msg: Dict[str, Any]) -> None:
        try:
            ft_dir = Path(str(msg.get("ft_run_dir") or msg.get("out_dir") or ""))
            reason = str(msg.get("reason") or "operator_confirmed")
            if self._active_sim_mode:
                from experiment_game.experiment.sim.sim_registry import (
                    login_sim_subject,
                    promote_sim_ft_to_current,
                    validate_sim_subject_id,
                )

                sid = validate_sim_subject_id(str(msg.get("subject_id") or self._active_subject or ""))
                prom = promote_sim_ft_to_current(
                    sid, ft_dir, repo_root=self.repo_root, reason=reason
                )
                if self._active_subject == sid:
                    self._active_subject_info = login_sim_subject(sid, repo_root=self.repo_root)
                    self._active_subject_info["sim_mode"] = True
            else:
                from experiment_game.experiment.subject_registry import (
                    login_subject,
                    promote_ft_to_current,
                    validate_subject_id,
                )

                sid = validate_subject_id(str(msg.get("subject_id") or self._active_subject or ""))
                prom = promote_ft_to_current(sid, ft_dir, repo_root=self.repo_root, reason=reason)
                if self._active_subject == sid:
                    self._active_subject_info = login_subject(sid, repo_root=self.repo_root)
                    self._active_subject_info["sim_mode"] = False
            self.bridge.broadcast(
                {
                    "type": "finetune_promote_ack",
                    "ok": True,
                    **prom,
                    **self._model_presets_payload(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {"type": "finetune_promote_ack", "ok": False, "message": str(exc)}
            )

    def _handle_model_eval_grid(self, msg: Dict[str, Any]) -> None:
        """需求 2026-08-30 二.4：历史模型（current + ft_runs）× v3 会话 识别结果。"""
        if getattr(self, "_eval_busy", False):
            self.bridge.broadcast({
                "type": "model_eval_ack", "ok": False,
                "message": "已有评测任务在运行",
            })
            return
        try:
            if self._active_sim_mode:
                raise ValueError("仿真模式暂不支持模型评测网格")
            from experiment_game.experiment.subject_registry import (
                list_sessions,
                models_current_dir,
                validate_subject_id,
            )

            sid = validate_subject_id(
                str(msg.get("subject_id") or self._active_subject or "")
            )
            paths_raw = msg.get("session_paths") or []
            if paths_raw:
                session_dirs = [Path(p) for p in paths_raw]
            else:
                session_dirs = [
                    Path(s["path"])
                    for s in list_sessions(sid, repo_root=self.repo_root)
                    if s.get("phase_mode") == "v3_session"
                ]
            if not session_dirs:
                self.bridge.broadcast({
                    "type": "model_eval_ack", "ok": False,
                    "message": "没有 v3 会话可评测",
                })
                return
            models_dir = models_current_dir(sid, repo_root=self.repo_root).parent
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast({"type": "model_eval_ack", "ok": False, "message": str(exc)})
            return

        def _job() -> None:
            self._eval_busy = True
            try:
                from experiment_game.pipeline.model_eval import evaluate_model_grid

                def _prog(stage: str, i: int) -> None:
                    self.bridge.broadcast({
                        "type": "model_eval_progress", "stage": stage, "index": int(i),
                    })

                self.bridge.broadcast({
                    "type": "model_eval_ack", "ok": True,
                    "message": f"开始评测 {len(session_dirs)} 个 v3 会话…",
                    "n_sessions": len(session_dirs),
                })
                result = evaluate_model_grid(
                    models_dir, session_dirs, on_progress=_prog,
                )
                result["subject_id"] = sid
                self.bridge.broadcast({"type": "model_eval_result", "ok": True, **result})
            except Exception as exc:  # noqa: BLE001
                self.bridge.broadcast({
                    "type": "model_eval_result", "ok": False, "message": str(exc),
                })
            finally:
                self._eval_busy = False

        import threading as _threading

        _threading.Thread(target=_job, daemon=True, name="model-eval-grid").start()

    def _handle_session_exclude_record(self, msg: Dict[str, Any]) -> None:
        try:
            from experiment_game.experiment.sim.campaign import load_campaign
            from experiment_game.experiment.sim.campaign_summary import exclude_session_from_records

            session_root = Path(str(msg.get("session_root") or msg.get("root") or ""))
            campaign = msg.get("campaign_manifest") or msg.get("campaign")
            if isinstance(campaign, dict) and campaign.get("manifest_path"):
                campaign = load_campaign(campaign["manifest_path"])
            elif msg.get("campaign_manifest_path"):
                campaign = load_campaign(msg["campaign_manifest_path"])
            else:
                campaign = None

            result = exclude_session_from_records(
                session_root,
                campaign_manifest=campaign,
                repo_root=self.repo_root,
                reason=str(msg.get("reason") or "operator_summary_exclude"),
            )
            payload: Dict[str, Any] = {
                "type": "session_exclude_record_ack",
                "ok": True,
                **result,
            }
            if result.get("sim_index") and self._active_subject == result.get("subject_id"):
                payload.update(self._model_presets_payload())
            self.bridge.broadcast(payload)
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {
                    "type": "session_exclude_record_ack",
                    "ok": False,
                    "message": str(exc),
                }
            )

    def _maybe_archive_overwrite_session(self, cfg: Dict[str, Any]) -> None:
        """若 experiment.overwrite_session_id：归档同 session_id 的旧目录。"""
        exp = cfg.get("experiment") or {}
        if not bool(exp.get("overwrite_session_id")):
            return
        if str(exp.get("phase_mode") or "") == "sim_v3_session":
            return
        sub = cfg.get("subject") or {}
        sid = str(sub.get("subject_id") or "").strip()
        sess = str(sub.get("session_id") or "").strip()
        if not sid or not sess:
            return
        from experiment_game.experiment.subject_registry import archive_sessions_for_id

        phase_mode = str(exp.get("phase_mode") or "").strip() or None
        moved = archive_sessions_for_id(
            sid,
            sess,
            repo_root=self.repo_root,
            phase_mode=phase_mode,
        )
        if moved:
            print(f"[operator] 已归档 {len(moved)} 个同板块同编号会话: {sess} ({phase_mode})")
            self.bridge.broadcast(
                {
                    "type": "session_overwrite_ack",
                    "ok": True,
                    "subject_id": sid,
                    "session_id": sess,
                    "phase_mode": phase_mode,
                    "archived": moved,
                    "message": f"已归档 {len(moved)} 个旧目录后重采 {sess}",
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
            try:
                self.bridge.clear_pending_session_saved()
            except Exception:  # noqa: BLE001
                pass

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
        self._layout_finalized = False
        err: Optional[BaseException] = None
        try:
            self._run_session(cfg)
        except Exception as exc:  # noqa: BLE001
            err = exc
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
            if not self._layout_finalized and self._paths is not None:
                try:
                    storage = (cfg or {}).get("storage") or {}
                    layout = str(storage.get("save_layout") or "phase_folders")
                    report = ensure_crash_artifacts(
                        self._paths.root,
                        aborted=bool(err),
                        reason=(
                            f"session_error:{type(err).__name__}"
                            if err
                            else "session_exit_without_layout_finalize"
                        ),
                        acq_enabled=bool((cfg.get("acquisition") or {}).get("enabled", True)),
                        save_layout=layout,
                        save_continuous=bool(storage.get("save_continuous_master", True)),
                        save_phase_slices=bool(
                            storage.get("save_phase_slices") or layout == "phase_folders"
                        ),
                    )
                    print(f"[operator] 异常收尾落盘: {report}", flush=True)
                    self._layout_finalized = True
                except Exception as fin_exc:  # noqa: BLE001
                    print(f"[operator] 异常收尾失败: {fin_exc}", file=sys.stderr)
            self._unregister_session_atexit()
            with self._lock:
                self._busy = False

    def _mark_layout_finalized(self) -> None:
        self._layout_finalized = True
        self._unregister_session_atexit()

    def _register_session_atexit(self) -> None:
        if self._crash_atexit_registered:
            return
        atexit.register(self._atexit_crash_finalize)
        self._crash_atexit_registered = True

    def _unregister_session_atexit(self) -> None:
        if not self._crash_atexit_registered:
            return
        try:
            atexit.unregister(self._atexit_crash_finalize)
        except Exception:  # noqa: BLE001
            pass
        self._crash_atexit_registered = False

    def _atexit_crash_finalize(self) -> None:
        if self._layout_finalized or self._paths is None:
            return
        try:
            cfg = self._last_config or {}
            storage = cfg.get("storage") or {}
            layout = str(storage.get("save_layout") or "phase_folders")
            ensure_crash_artifacts(
                self._paths.root,
                aborted=True,
                reason="process_atexit",
                acq_enabled=bool((cfg.get("acquisition") or {}).get("enabled", True)),
                save_layout=layout,
                save_continuous=bool(storage.get("save_continuous_master", True)),
                save_phase_slices=bool(
                    storage.get("save_phase_slices") or layout == "phase_folders"
                ),
            )
        except Exception:  # noqa: BLE001
            pass

    def _bind_session_paths(self, paths: SessionPaths, cfg: Dict[str, Any]) -> None:
        self._paths = paths
        self._last_config = cfg
        self._layout_finalized = False
        self._register_session_atexit()

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
        self._maybe_archive_overwrite_session(cfg)
        if phase_mode == "v3_session":
            self._run_v3_session(cfg)
            return
        if phase_mode == "sim_v3_session":
            self._run_sim_v3_session(cfg)
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

            paths = create_session_dir(save_root, subject_id, session_id, module_prefix="v1")
            self._bind_session_paths(paths, cfg)

            # 会话快照（便于复现；UI-2 正式化）。继续段记录该段实际 trial 数
            seg_cfg = dict(cfg)
            seg_cfg = {
                **cfg,
                "subject": {**sub, "session_id": session_id},
                "experiment": {**exp, "acquire_trials": remaining},
            }
            atomic_write_json(paths.root / "run_config.json", seg_cfg)
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
                    "phase_mode": "phase2_full",
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
                self._mark_layout_finalized()
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
                phase4_result = self._resolve_phase4_runner()(
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

        paths = create_session_dir(save_root, subject_id, session_id, module_prefix="v2")
        self._bind_session_paths(paths, cfg)
        atomic_write_json(paths.root / "run_config.json", cfg)
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
        # v3 权重游戏测试：注入被试模型目录（current/members + overlay = v3 最终权重）
        if getattr(v2_cfg, "use_v3_weights", False) and not getattr(v2_cfg, "subject_models_dir", ""):
            from experiment_game.experiment.subject_registry import subject_root

            v2_cfg.subject_models_dir = str(
                subject_root(subject_id, repo_root=self.repo_root) / "models"
            )
        verr = v2_cfg.verify_errors()
        if verr:
            raise RuntimeError("v2 配置无效: " + "; ".join(verr))
        from experiment_game.experiment.session_v2 import (
            diagnose_v2_online_deps,
            probe_v2_weights_missing,
        )

        weights_missing = probe_v2_weights_missing(v2_cfg)
        skip_cal = bool(exp.get("skip_v2_calibration"))
        skip_game = bool(exp.get("skip_v2_game"))
        if getattr(v2_cfg, "game_mode", "") == "v3_test":
            v2_score_max = round(
                int(v2_cfg.v3test_n_rest) * float(v2_cfg.v3test_rest_points)
                + (int(v2_cfg.v3test_n_left) + int(v2_cfg.v3test_n_right))
                * float(v2_cfg.v3test_mi_points),
                2,
            )
        else:
            v2_score_max = 0
            if not skip_cal:
                v2_score_max += int(v2_cfg.cal_rounds_max) * int(v2_cfg.trials_per_round)
            if not skip_game:
                v2_score_max += int(v2_cfg.game_rounds) * int(v2_cfg.game_trials_per_round)
            v2_score_max = session_score_max_openbmi(
                v2_score_max, inter_trial_rest_s=float(v2_cfg.inter_trial_rest_s)
            )
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
            "subject_feedback_mode": _subject_feedback_mode(cfg),
            "weights": self._weights_from_cfg(v2_cfg),
                "timing": {
                    "prep_s": v2_cfg.prep_s,
                    "cue_s": v2_cfg.cue_s,
                    "mi_s": v2_cfg.imagine_s,
                    "iti_s": v2_cfg.iti_s,
                    "inter_trial_rest_s": v2_cfg.inter_trial_rest_s,
                    "fixation_s": v2_cfg.prep_s,
                    "post_mi_hold_s": 0,
                    # rest_s = Cue前试次间 Rest（与操作台时间轴 rest_s 键对齐）
                    "rest_s": v2_cfg.inter_trial_rest_s,
                    "transition_s": v2_cfg.iti_s,
                },
                "trial_total_s": v2_cfg.trial_total_s(),
                "session_score": 0,
                "session_score_max": v2_score_max,
                "session_trials_done": 0,
                "v2_config_effective": {
                    "cal_rounds_min": v2_cfg.cal_rounds_min,
                    "cal_rounds_max": v2_cfg.cal_rounds_max,
                    "game_rounds": v2_cfg.game_rounds,
                    "gate_enter_three": v2_cfg.gate_enter_three,
                    "s3_task_ckpt": v2_cfg.s3_task_ckpt,
                    "s3_three_ckpt": v2_cfg.s3_three_ckpt,
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
                live_buf = self._live_capture.buf if self._live_capture is not None else None
                deps, reasons = diagnose_v2_online_deps(
                    v2_cfg,
                    require_lsl=True,
                    lsl_timeout_s=8.0,
                    on_console=print,
                    buffer=live_buf,
                )
                online_ok = deps is not None
                self._v2_injected_deps = deps if online_ok else None
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
            subject_feedback_mode=_subject_feedback_mode(cfg),
            deps=getattr(self, "_v2_injected_deps", None),
            close_buffer=False,
        )
        self._v2_injected_deps = None
        self._runner = None

        events.emit("session_end", subject_id=subject_id, session_id=session_id, phase="v2_session")
        markers.push("session_end|phase=v2_session")

        self._shutdown_session_resources()

        self.bridge.broadcast({
            "type": "session_finishing",
            "phase_mode": "v2_session",
            "subject_id": subject_id,
            "root": str(paths.root),
            "message": "v2 试次已结束，正在落盘…",
        })

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
            self._mark_layout_finalized()
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
                run_p4_cal, run_p4_game = self._resolve_phase4_v2_pair()

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

        paths = create_session_dir(save_root, subject_id, session_id, module_prefix="v3")
        self._bind_session_paths(paths, cfg)
        atomic_write_json(paths.root / "run_config.json", cfg)
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
            "subject_feedback_mode": _subject_feedback_mode(cfg),
            "v3_block_order": b_order,
            "weights": self._weights_from_cfg(v3_cfg),
            "timing": {
                "prep_s": v3_cfg.prep_s,
                "cue_s": v3_cfg.cue_s,
                "mi_s": v3_cfg.imagine_s,
                "iti_s": v3_cfg.iti_s,
                "inter_trial_rest_s": v3_cfg.inter_trial_rest_s,
                "fixation_s": v3_cfg.prep_s,
                "post_mi_hold_s": 0,
                # rest_s = Cue前试次间 Rest（与操作台时间轴 rest_s 键对齐）
                "rest_s": v3_cfg.inter_trial_rest_s,
                "transition_s": v3_cfg.iti_s,
            },
            "trial_total_s": v3_cfg.trial_total_s(),
            "session_score": 0,
            "session_score_max": session_score_max_openbmi(
                int(v3_cfg.blocks) * int(v3_cfg.trials_per_block),
                inter_trial_rest_s=float(v3_cfg.inter_trial_rest_s),
            ),
            "session_trials_done": 0,
            "v3_config_effective": {
                "blocks": v3_cfg.blocks,
                "trials_per_block": v3_cfg.trials_per_block,
                "baseline_rest_s": v3_cfg.baseline_rest_s,
                "s3_task_ckpt": v3_cfg.s3_task_ckpt,
                "s3_three_ckpt": v3_cfg.s3_three_ckpt,
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
            from experiment_game.experiment.session_v2 import diagnose_v2_online_deps
            live_buf = self._live_capture.buf if self._live_capture is not None else None
            deps, reasons = diagnose_v2_online_deps(
                v3_cfg,  # type: ignore[arg-type]
                require_lsl=True,
                lsl_timeout_s=8.0,
                on_console=print,
                buffer=live_buf,
            )
            if deps is None:
                msg = (
                    "v3 自检失败（权重或 LSL 不可用）："
                    + ("；".join(reasons) if reasons else "未知")
                    + "\n请确认 open_operator.bat、采集已开、ckpt 路径正确。"
                )
                raise RuntimeError(msg)
            self._v3_injected_deps = deps
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
            subject_feedback_mode=_subject_feedback_mode(cfg),
            use_synthetic=use_synthetic,
            deps=getattr(self, "_v3_injected_deps", None),
            close_buffer=False,
        )
        self._v3_injected_deps = None
        self._runner = None

        events.emit("session_end", subject_id=subject_id, session_id=session_id, phase="v3_session")
        markers.push("session_end|phase=v3_session")
        self._shutdown_session_resources()

        self.bridge.broadcast({
            "type": "session_finishing",
            "phase_mode": "v3_session",
            "subject_id": subject_id,
            "root": str(paths.root),
            "v3_summary": {
                "session_score": (summary or {}).get("session_score"),
                "session_score_max": (summary or {}).get("session_score_max"),
                "session_score_by": (summary or {}).get("session_score_by"),
                "session_trials_done": (summary or {}).get("session_trials_done"),
                "window_acc": (summary or {}).get("window_acc"),
                "window_acc_n": (summary or {}).get("window_acc_n"),
                "quality_tier": (summary or {}).get("quality_tier"),
                "n_trials": (summary or {}).get("n_trials"),
                "frozen": (summary or {}).get("frozen"),
            },
            "message": "v3 试次已结束，正在落盘…",
        })
        self._emit_acq_status("stopped", "录制已停止 · 落盘中")

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
            self._mark_layout_finalized()
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
        suggest_sid = None
        try:
            from experiment_game.experiment.subject_registry import build_index

            idx = build_index(subject_id, repo_root=self.repo_root)
            suggest_sid = idx.get("suggest_session_id")
        except Exception:  # noqa: BLE001
            pass
        self.bridge.broadcast({
            "type": "session_saved",
            "root": str(paths.root),
            "files": files,
            "acq_enabled": acq_on,
            "train_eligible": acq_on and not bool((summary or {}).get("aborted")),
            "verify": verify,
            "v3_summary": summary,
            "phase_mode": "v3_session",
            "subject_id": subject_id,
            "suggest_session_id": suggest_sid,
            "message": (
                "v3 会话已中止（动觉引导超时或其它中止）"
                if (summary or {}).get("aborted")
                else "v3 探针会话已结束"
            ),
        })
        self.bridge.broadcast({"type": "session", "status": "done"})
        self._emit_acq_status("stopped", "录制已停止")
        print(f"[operator] v3 会话目录: {paths.root}")

    def _handle_sim_catalog(self, msg: Dict[str, Any]) -> None:
        from experiment_game.experiment.sim.bci2a_catalog import list_subject_runs
        from experiment_game.experiment.sim.sim_registry import validate_sim_subject_id

        try:
            sid = validate_sim_subject_id(str(msg.get("subject_id") or self._active_subject or ""))
            runs = list_subject_runs(sid)
            self.bridge.broadcast(
                {"type": "sim_catalog_ack", "ok": True, "subject_id": sid, "runs": runs}
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {"type": "sim_catalog_ack", "ok": False, "message": str(exc)}
            )

    def _handle_sim_campaign_create(self, msg: Dict[str, Any]) -> None:
        from experiment_game.experiment.sim.bci2a_catalog import resolve_mat_path
        from experiment_game.experiment.sim.bci2a_mat_loader import count_run_capacity, load_bci2a_run
        from experiment_game.experiment.sim.campaign import create_campaign
        from experiment_game.experiment.sim.sim_registry import validate_sim_subject_id

        try:
            sid = validate_sim_subject_id(str(msg.get("subject_id") or self._active_subject or ""))
            queue = msg.get("session_queue") or msg.get("runs") or []
            n_trials = int(msg.get("session_trials_total") or 36)
            mat_path = resolve_mat_path(sid)
            for rid in queue:
                rd = load_bci2a_run(mat_path, str(rid).strip().lower())
                _, _, _, n_max = count_run_capacity(rd)
                if n_trials > n_max:
                    raise ValueError(
                        f"run {rid} 最多 {n_max} 试次（Rest+L/R），当前 {n_trials}"
                    )
            manifest = create_campaign(
                sid,
                list(queue),
                session_trials_total=int(msg.get("session_trials_total") or 36),
                replay_align=str(msg.get("replay_align") or "schedule_align"),
                replay_speed=float(msg.get("replay_speed") or 4.0),
                leave_next_mode=bool(msg.get("leave_next_mode", True)),
                repo_root=self.repo_root,
            )
            self.bridge.broadcast(
                {"type": "sim_campaign_ack", "ok": True, "manifest": manifest}
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {"type": "sim_campaign_ack", "ok": False, "message": str(exc)}
            )

    def _handle_sim_campaign_list(self, msg: Dict[str, Any]) -> None:
        from experiment_game.experiment.sim.sim_registry import list_campaigns, validate_sim_subject_id

        try:
            sid = validate_sim_subject_id(str(msg.get("subject_id") or self._active_subject or ""))
            campaigns = list_campaigns(sid, repo_root=self.repo_root)
            self.bridge.broadcast(
                {"type": "sim_campaign_list_ack", "ok": True, "subject_id": sid, "campaigns": campaigns}
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {"type": "sim_campaign_list_ack", "ok": False, "message": str(exc)}
            )

    def _handle_ramp_status(self, msg: Dict[str, Any]) -> None:
        from experiment_game.experiment.sim.campaign import load_campaign
        from experiment_game.experiment.sim.ramp import (
            ft_replay_recommendation,
            leave_next_train_runs,
            next_eval_run,
            ramp_stage,
        )
        from experiment_game.experiment.sim.sim_registry import validate_sim_subject_id

        try:
            sid = validate_sim_subject_id(str(msg.get("subject_id") or self._active_subject or ""))
            manifest_path = msg.get("campaign_manifest")
            eval_run = str(msg.get("eval_run_id") or "").strip().lower()
            if not manifest_path:
                raise ValueError("缺少 campaign_manifest")
            manifest = load_campaign(manifest_path)
            if not eval_run:
                eval_run = str(next_eval_run(manifest) or "").strip().lower()
                if not eval_run:
                    done = manifest.get("sessions_completed") or []
                    if done:
                        eval_run = str(done[-1].get("run_id") or "").strip().lower()
            stage = ramp_stage(manifest, eval_run) if eval_run else 0
            train = leave_next_train_runs(manifest, eval_run) if eval_run else []
            self.bridge.broadcast(
                {
                    "type": "ramp_status_ack",
                    "ok": True,
                    "subject_id": sid,
                    "campaign_id": manifest.get("campaign_id"),
                    "eval_run_id": eval_run or None,
                    "ramp_stage": stage,
                    "leave_next_train": [{"run_id": rid, "session_dir": p} for rid, p in train],
                    "ft_replay_recommendation": ft_replay_recommendation(stage),
                    "leave_next_mode": bool(manifest.get("leave_next_mode", True)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.bridge.broadcast(
                {"type": "ramp_status_ack", "ok": False, "message": str(exc)}
            )

    def _run_sim_v3_session(self, cfg: Dict[str, Any]) -> None:
        """BCI2a 仿真 v3：回放 mat run，跳过 baseline/gap。"""
        sub = cfg["subject"]
        acq_cfg = cfg["acquisition"]
        exp = cfg["experiment"]
        storage = cfg["storage"]
        sim_ext = (cfg.get("extensions") or {}).get("sim") or {}
        if not isinstance(sim_ext, dict):
            sim_ext = {}

        subject_id = str(sub["subject_id"]).upper()
        run_id = str(sim_ext.get("run_id") or sub.get("session_id") or "").strip().lower()
        use_campaign_queue = bool(sim_ext.get("use_campaign_queue"))

        from experiment_game.experiment.sim.bci2a_catalog import resolve_mat_path
        from experiment_game.experiment.sim.bci2a_replay_source import Bci2aReplaySource
        from experiment_game.experiment.sim.campaign import load_campaign, pop_next_run, save_campaign
        from experiment_game.experiment.sim.campaign_summary import append_session_result, write_campaign_summary
        from experiment_game.experiment.sim.run_to_session_map import build_sim_script
        from experiment_game.experiment.sim.sim_script_io import write_sim_script
        from experiment_game.experiment.sim.sim_registry import build_sim_index, storage_paths_for_sim
        from experiment_game.experiment.v3_config import V3Config
        from experiment_game.experiment.session_v3 import build_v3_deps_from_buffer
        from experiment_game.experiment.inference_v2 import RingBuffer

        save_root = Path(storage.get("save_root") or storage_paths_for_sim(subject_id, repo_root=self.repo_root)["save_root"])
        if not save_root.is_absolute():
            save_root = (self.repo_root / save_root).resolve()

        campaign_manifest = None
        manifest_path = sim_ext.get("campaign_manifest")
        if manifest_path:
            campaign_manifest = load_campaign(manifest_path)
            if use_campaign_queue or not run_id:
                nxt = pop_next_run(campaign_manifest)
                if not nxt:
                    raise RuntimeError("Campaign 队列已空")
                run_id = nxt
            else:
                consumed = set(campaign_manifest.get("runs_consumed") or [])
                if run_id in consumed:
                    raise RuntimeError(f"run {run_id} 已在 Campaign 中使用")

        if not run_id.startswith("run"):
            raise RuntimeError("仿真须指定 run_id（如 run3）或启用 Campaign 队列")

        protocol_locked = bool(exp.get("protocol_locked", True))
        v3_overrides = dict(exp.get("v3_overrides") or {}) if isinstance(exp.get("v3_overrides"), dict) else {}
        v3_path = exp.get("v3_config_path")
        v3_cfg = V3Config.load_yaml(v3_path) if v3_path else V3Config.load_yaml()
        v3_cfg.apply_overrides(v3_overrides, protocol_locked=protocol_locked)

        session_trials_total = int(
            sim_ext.get("session_trials_total")
            or (campaign_manifest or {}).get("session_trials_total")
            or 36
        )
        replay_speed = float(
            sim_ext.get("replay_speed")
            or (campaign_manifest or {}).get("replay_speed")
            or 4.0
        )
        replay_align = str(
            sim_ext.get("replay_align")
            or (campaign_manifest or {}).get("replay_align")
            or "schedule_align"
        )
        mat_path = resolve_mat_path(subject_id)
        # 先校验 run 容量（Rest+L+R），避免落盘空 session 目录
        script = build_sim_script(
            mat_path,
            run_id,
            session_trials_total=session_trials_total,
            blocks=int(v3_cfg.blocks),
            align_mode=replay_align,
            seed=exp.get("seed"),
            rest_s=float(v3_cfg.inter_trial_rest_s),
        )

        session_id = run_id
        paths = create_session_dir(save_root, subject_id, session_id, module_prefix="sim")
        self._bind_session_paths(paths, cfg)
        write_sim_script(paths.root, script)
        atomic_write_json(paths.root / "run_config.json", cfg)
        self._maybe_persist_last_config(cfg)

        v3_cfg.blocks = script.blocks
        v3_cfg.trials_per_block = script.trials_per_block
        verr = v3_cfg.verify_errors()
        if verr:
            raise RuntimeError("v3 配置无效: " + "; ".join(verr))

        seed = exp.get("seed")
        b_order = [f"sim_b{i + 1}" for i in range(script.blocks)]

        meta = SessionMeta(
            subject_id=subject_id,
            session_id=session_id,
            phase_mode="sim_v3_session",
            use_synthetic=False,
            trial_count=script.session_trials_total,
            object="cup",
            scene="home_desk",
            notes=str(sub.get("notes") or "bci2a_sim_replay"),
            channel_labels=list(acq_cfg.get("channel_labels") or DEFAULT_CHANNEL_LABELS),
        )
        write_session_meta(paths.meta_json, meta)
        sim_meta = {
            "sim_mode": True,
            "source_mat": str(mat_path),
            "source_run": run_id,
            "campaign_id": (campaign_manifest or {}).get("campaign_id"),
            "session_trials_total": script.session_trials_total,
            "trials_unused": script.trials_unused,
            "three_class": script.meta.get("three_class"),
            "n_rest_used": script.meta.get("n_rest_used"),
            "n_left_used": script.meta.get("n_left_used"),
            "n_right_used": script.meta.get("n_right_used"),
            "n_total_available": script.meta.get("n_total_available"),
            "skip_session_baseline": True,
            "skip_block_gap": True,
            "replay_speed": replay_speed,
            "replay_align": replay_align,
        }
        update_session_meta(
            paths.meta_json,
            v3_config=str(v3_path or "config/v3_session.yaml"),
            v3_config_effective=v3_cfg.to_dict(),
            protocol_locked=protocol_locked,
            v3_block_order=b_order,
            v3_seed=seed,
            **sim_meta,
        )

        self.bridge.broadcast({
            "type": "session_started",
            "session_root": str(paths.root),
            "subject_url": self.subject_url,
            "acq_enabled": True,
            "board_mode": "bci2a_replay",
            "phase_mode": "sim_v3_session",
            "degraded": False,
            "save_root": str(save_root),
            "open_subject_page": exp.get("open_subject_page", True),
            "segment": 1,
            "protocol_locked": protocol_locked,
            "subject_feedback_mode": _subject_feedback_mode(cfg),
            "v3_block_order": b_order,
            "weights": self._weights_from_cfg(v3_cfg),
            "timing": {
                "prep_s": v3_cfg.prep_s,
                "cue_s": v3_cfg.cue_s,
                "mi_s": v3_cfg.imagine_s,
                "iti_s": v3_cfg.iti_s,
                "inter_trial_rest_s": v3_cfg.inter_trial_rest_s,
                "fixation_s": v3_cfg.prep_s,
                "post_mi_hold_s": 0,
                "rest_s": v3_cfg.inter_trial_rest_s,
                "transition_s": v3_cfg.iti_s,
            },
            "trial_total_s": v3_cfg.trial_total_s(),
            "session_score": 0,
            "session_score_max": session_score_max_openbmi(
                int(script.session_trials_total),
                inter_trial_rest_s=float(v3_cfg.inter_trial_rest_s),
            ),
            "session_trials_done": 0,
            "sim": sim_meta,
            "v3_config_effective": {
                "blocks": v3_cfg.blocks,
                "trials_per_block": v3_cfg.trials_per_block,
                "baseline_rest_s": v3_cfg.baseline_rest_s,
                "s3_task_ckpt": v3_cfg.s3_task_ckpt,
                "s3_three_ckpt": v3_cfg.s3_three_ckpt,
            },
        })

        events = EventLogger(paths.events_jsonl)
        self._events = events
        markers = MarkerPublisher(enabled=bool(acq_cfg.get("markers_lsl", True)))
        self._markers = markers
        runner = SessionRunner(events, markers, self.bridge, on_console=print)
        self._runner = runner

        buf = RingBuffer(capacity_s=max(120.0, script.session_trials_total * 20.0))
        buf.push_rate_scale = replay_speed
        replay: Optional[Bci2aReplaySource] = None
        try:
            deps = build_v3_deps_from_buffer(v3_cfg, buf, on_console=print)
            replay = Bci2aReplaySource(
                script,
                buf,
                eeg_csv_path=paths.eeg_csv,
                speed=replay_speed,
                align_mode=replay_align,
                rest_s=float(v3_cfg.inter_trial_rest_s),
                prep_s=float(v3_cfg.prep_s),
                mi_s=float(v3_cfg.imagine_s),
                iti_s=float(v3_cfg.iti_s),
            )
            self._emit_acq_status("connecting", f"仿真回放 {subject_id} · {run_id}…")
            replay.start()
            self._emit_acq_status("recording", f"仿真回放中 · {replay_speed}×")

            events.emit("session_start", subject_id=subject_id, session_id=session_id, phase="sim_v3_session")
            markers.push(f"session_start|subject={subject_id}|session={session_id}|phase=sim_v3_session|run={run_id}")
            self.bridge.broadcast({"type": "session", "status": "running", "phase": "sim_v3_session"})

            timeout = float(exp.get("ready_timeout_s") or 90)
            print(f"[operator] 等待诱导页 ready（{timeout:.0f}s）…")
            runner.wait_browser_ready(timeout=timeout)

            summary = runner.run_v3_session(
                config_path=v3_path,
                v3_overrides={
                    **v3_overrides,
                    "blocks": script.blocks,
                    "trials_per_block": script.trials_per_block,
                },
                protocol_locked=protocol_locked,
                seed=seed,
                subject_id=subject_id,
                subject_feedback_mode=_subject_feedback_mode(cfg),
                deps=deps,
                skip_session_baseline=True,
                skip_block_gap=True,
                block_order_override=b_order,
                trial_labels_by_block=script.labels_by_block,
                sim_meta=sim_meta,
                auto_confirm_guidance=True,
            )
        finally:
            if replay is not None:
                replay.stop()
            self._runner = None

        events.emit("session_end", subject_id=subject_id, session_id=session_id, phase="sim_v3_session")
        markers.push("session_end|phase=sim_v3_session")
        self._shutdown_session_resources()

        # 先通知前端离场，再做耗时落盘/对齐（避免静默期断线丢 session_saved）
        self.bridge.broadcast({
            "type": "session_finishing",
            "phase_mode": "sim_v3_session",
            "subject_id": subject_id,
            "root": str(paths.root),
            "v3_summary": {
                "session_score": (summary or {}).get("session_score"),
                "session_score_max": (summary or {}).get("session_score_max"),
                "session_score_by": (summary or {}).get("session_score_by"),
                "session_trials_done": (summary or {}).get("session_trials_done"),
                "window_acc": (summary or {}).get("window_acc"),
                "window_acc_n": (summary or {}).get("window_acc_n"),
                "quality_tier": (summary or {}).get("quality_tier"),
                "n_trials": (summary or {}).get("n_trials"),
                "frozen": (summary or {}).get("frozen"),
            },
            "message": f"仿真试次已结束，正在落盘… · {run_id}",
        })
        self._emit_acq_status("stopped", "仿真回放已停止 · 落盘中")

        layout = str(storage.get("save_layout") or "phase_folders")
        try:
            finalize_session_layout(
                paths.root,
                save_layout=layout,
                save_continuous=bool(storage.get("save_continuous_master", True)),
                save_phase_slices=bool(
                    storage.get("save_phase_slices") or layout == "phase_folders"
                ),
                acq_enabled=True,
            )
            self._mark_layout_finalized()
        except Exception as exc:  # noqa: BLE001
            print(f"[operator] 整理落盘目录失败: {exc}", file=sys.stderr)

        verify = {}
        try:
            verify = write_alignment_bundle(paths.root, acq_enabled=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[operator] alignment 失败: {exc}", file=sys.stderr)
            verify = {"passed": False, "errors": [str(exc)]}

        campaign_summary_path = None
        campaign_next_run = None
        if campaign_manifest is not None:
            if not use_campaign_queue:
                consumed = set(campaign_manifest.get("runs_consumed") or [])
                if run_id not in consumed:
                    consumed.add(run_id)
                    campaign_manifest["runs_consumed"] = sorted(consumed)
                    save_campaign(campaign_manifest)
            append_session_result(
                campaign_manifest,
                session_dir=paths.root,
                summary=summary,
            )
            campaign_summary_path = str(
                write_campaign_summary(campaign_manifest, repo_root=self.repo_root)
            )
            campaign_manifest = load_campaign(campaign_manifest["manifest_path"])
            remaining = [
                r
                for r in (campaign_manifest.get("session_queue") or [])
                if r not in set(campaign_manifest.get("runs_consumed") or [])
            ]
            campaign_next_run = remaining[0] if remaining else None

        sim_index = None
        try:
            sim_index = build_sim_index(subject_id, repo_root=self.repo_root)
        except Exception:  # noqa: BLE001
            pass

        update_session_meta(paths.meta_json, v3_summary=summary, **sim_meta)
        files = self._list_session_files(paths.root)
        saved_payload: Dict[str, Any] = {
            "type": "session_saved",
            "root": str(paths.root),
            "files": files,
            "acq_enabled": True,
            "train_eligible": True,
            "verify": verify,
            "v3_summary": summary,
            "phase_mode": "sim_v3_session",
            "subject_id": subject_id,
            "sim": sim_meta,
            "message": f"仿真 session 已结束 · {run_id}",
        }
        if campaign_manifest is not None:
            saved_payload["campaign"] = {
                "manifest": campaign_manifest,
                "summary_path": campaign_summary_path,
                "next_run": campaign_next_run,
                "completed": campaign_manifest.get("status") == "completed"
                or campaign_next_run is None,
            }
        if sim_index is not None:
            saved_payload["sim_index"] = sim_index
            saved_payload["suggest_session_id"] = sim_index.get("suggest_session_id")
        if self._active_subject == subject_id:
            saved_payload.update(self._model_presets_payload())
        self.bridge.broadcast(saved_payload)
        self.bridge.broadcast({"type": "session", "status": "done"})
        self._emit_acq_status("stopped", "仿真回放已停止")
        print(f"[operator] 仿真会话目录: {paths.root}")

    def _run_v4_session(self, cfg: Dict[str, Any]) -> None:
        """v4 实验前数据质量检测：无模型、连续帽检。"""
        sub = cfg["subject"]
        acq_cfg = cfg["acquisition"]
        exp = cfg["experiment"]
        storage = cfg["storage"]
        save_root = Path(storage["save_root"])
        subject_id = sub["subject_id"]
        session_id = sub["session_id"]

        paths = create_session_dir(save_root, subject_id, session_id, module_prefix="v4")
        self._bind_session_paths(paths, cfg)
        atomic_write_json(paths.root / "run_config.json", cfg)
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
            "open_subject_page": exp.get("open_subject_page", True),
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
            if self._live_capture is None:
                raise RuntimeError("v4 LiveEegCapture 未启动")
            buf = self._live_capture.buf
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
            pass  # live_capture 由 _shutdown_session_resources 关闭
        self._runner = None

        events.emit("session_end", subject_id=subject_id, session_id=session_id, phase="v4_session")
        markers.push("session_end|phase=v4_session")
        self._shutdown_session_resources()

        self.bridge.broadcast({
            "type": "session_finishing",
            "phase_mode": "v4_session",
            "subject_id": subject_id,
            "root": str(paths.root),
            "message": "v4 检测已结束，正在落盘…",
        })

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
            self._mark_layout_finalized()
        except Exception as exc:  # noqa: BLE001
            print(f"[operator] 整理落盘目录失败: {exc}", file=sys.stderr)

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
                    run_p4_cal, run_p4_game = self._resolve_phase4_v2_pair()
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
                result = self._resolve_phase4_runner()(
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
        live = self._live_capture
        self._live_capture = None
        if live is not None:
            try:
                meta = live.stop()
                quality = meta.get("quality") if isinstance(meta, dict) else None
                if isinstance(quality, dict) and quality:
                    self._last_acq_quality = quality
                    print(
                        f"[operator] Bus CSV 停止: rows={meta.get('samples_written')} "
                        f"drop_rate_pct={quality.get('drop_rate_pct')} "
                        f"timeline={quality.get('timeline', '?')}"
                    )
                else:
                    rows = meta.get("samples_written") if isinstance(meta, dict) else "?"
                    print(f"[operator] Bus CSV 停止: rows={rows}")
            except Exception as exc:  # noqa: BLE001
                print(f"[operator] Bus CSV 停止异常: {exc}", file=sys.stderr)

        acq = self._acq
        self._acq = None
        if acq is not None:
            try:
                report = acq.stop()
                print(f"[operator] 采集停止: {report.get('message')}")
                quality = report.get("quality")
                if isinstance(quality, dict) and quality and not self._last_acq_quality:
                    self._last_acq_quality = quality
            except Exception as exc:  # noqa: BLE001
                print(f"[operator] 停止采集异常: {exc}", file=sys.stderr)
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
            filter_enabled=bool(filt.get("enabled", False)),
            bandpass_low_hz=float(filt.get("bandpass_low_hz", 0.5)),
            bandpass_high_hz=float(filt.get("bandpass_high_hz", 45.0)),
            notch_enabled=bool(filt.get("notch_enabled", False)),
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
        # Bus CSV 单写：板卡只推 LSL，不启 lsl_connect Recorder
        self._acq.start(paths.eeg_csv, record_csv=False)
        self._acq.health_check()
        from experiment_game.experiment.live_capture import LiveEegCapture

        self._live_capture = LiveEegCapture(
            paths.eeg_csv,
            channel_labels=list(meta.channel_labels or DEFAULT_CHANNEL_LABELS),
            use_synthetic=use_synthetic,
            serial_port=str(acq_cfg.get("serial_port") or ""),
            sample_rate_hz=float(acq_cfg.get("sample_rate_hz") or 250),
        )
        self._live_capture.start(lsl_timeout_s=8.0)
        print("[operator] EEGBus CSV 订户已挂接（替代 Recorder 双轨）")
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
