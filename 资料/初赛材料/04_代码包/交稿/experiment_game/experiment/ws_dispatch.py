"""WS 入站 dispatch（总册 W6：从 orchestrator 抽出路由表）。

``OperatorService`` 仍持有业务 handler；本模块集中路由表与轻量 ack 逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from experiment_game.experiment.orchestrator import OperatorService

WsHandler = Callable[[Dict[str, Any]], None]
_PKG_ROOT = Path(__file__).resolve().parents[1]


def build_ws_dispatch_table(svc: "OperatorService") -> Dict[str, WsHandler]:
    """构造 type → handler 表（与现网行为一致）。"""
    return {
        "config_validate": lambda m: ws_config_validate(svc, m),
        "session_start": lambda m: svc._handle_session_start(m.get("run_config") or {}),
        "open_folder": lambda m: svc._open_folder(str(m.get("path") or "")),
        "open_subject_page": lambda m: svc._open_subject_page(),
        "list_serial_ports": lambda m: svc._list_serial_ports(),
        "save_defaults": lambda m: svc._save_defaults(m.get("run_config") or {}),
        "run_phase4": lambda m: svc._handle_run_phase4(str(m.get("path") or ""), m),
        "questionnaire_open": lambda m: svc._handle_questionnaire_open(),
        "questionnaire_result": svc._handle_questionnaire_result,
        "client_stats": svc._handle_client_stats,
        "subject_login": svc._handle_subject_login,
        "subject_logout": lambda m: svc._handle_subject_logout(),
        "subject_info": svc._handle_subject_info,
        "finetune_start": svc._handle_finetune_start,
        "finetune_promote": svc._handle_finetune_promote,
        "model_eval_grid": svc._handle_model_eval_grid,
        "session_exclude_record": svc._handle_session_exclude_record,
        "ramp_status": svc._handle_ramp_status,
        "sim_catalog": svc._handle_sim_catalog,
        "sim_campaign_create": svc._handle_sim_campaign_create,
        "sim_campaign_list": svc._handle_sim_campaign_list,
        "operator_hello": lambda m: ws_operator_hello(svc, m),
    }


def ws_config_validate(svc: "OperatorService", msg: Dict[str, Any]) -> None:
    from experiment_game.experiment.run_config import validate_run_config

    cfg, errors = validate_run_config(
        msg.get("run_config") or {},
        repo_root=svc.repo_root,
    )
    svc.bridge.broadcast(
        {
            "type": "config_ack",
            "ok": not errors,
            "errors": errors,
            "run_config": cfg if not errors else None,
        }
    )


def ws_operator_hello(svc: "OperatorService", msg: Dict[str, Any]) -> None:
    from experiment_game.experiment.defaults_store import defaults_path, load_operator_defaults
    from experiment_game.experiment.run_config import merge_run_config
    from experiment_game.experiment.serial_ports import list_serial_ports

    file_defaults, warn = load_operator_defaults(
        defaults_path(repo_pkg=_PKG_ROOT),
        repo_root=svc.repo_root,
    )
    svc.bridge.broadcast(
        {
            "type": "operator_hello",
            "message": "operator_connected",
            "operator_url": svc.operator_url,
            "subject_url": svc.subject_url,
            "defaults": file_defaults,
            "builtin_defaults": merge_run_config(None),
            "defaults_path": str(defaults_path(repo_pkg=_PKG_ROOT)),
            "defaults_warning": warn,
            "serial_ports": list_serial_ports(),
            "active_subject": svc._active_subject,
            "active_subject_info": svc._active_subject_info,
            **svc._model_presets_payload(),
        }
    )
