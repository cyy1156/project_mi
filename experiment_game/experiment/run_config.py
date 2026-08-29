"""操作台 run_config schema 校验与默认值（UI-1）。"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_game.core.channel_layout import DEFAULT_CHANNEL_LABELS
from experiment_game.core.channel_layout import DEVICE_CHANNEL_LABELS
from experiment_game.experiment.timing import timing_from_dict, validate_timing_dict

_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")

DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "schema_version": 2,
    "subject": {
        "subject_id": "sub01",
        "session_id": "ses01",
        "notes": "",
    },
    "acquisition": {
        "enabled": True,
        "board_mode": "synthetic",
        "serial_port": "COM5",
        "sample_rate_hz": 250,
        "channel_labels": list(DEFAULT_CHANNEL_LABELS),
        "filter": {
            "enabled": False,
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 45.0,
            "notch_enabled": False,
            "notch_low_hz": 49.0,
            "notch_high_hz": 51.0,
        },
        "markers_lsl": True,
    },
    "experiment": {
        "phase_mode": "phase2_full",
        "v2_config_path": None,
        "protocol_locked": True,
        "v2_overrides": {},
        "skip_v2_guidance": False,
        "skip_v2_calibration": False,
        "skip_v2_gate": False,
        "skip_v2_game": False,
        "acquire_trials": 8,
        "learn_trials_per_step": 2,
        "skip_adapt": False,
        "skip_learn": False,
        "skip_gate": False,
        "seed": None,
        "open_subject_page": True,
        "ready_timeout_s": 90,
        "timing": {
            "fixation_s": 2.0,
            "cue_s": 2.0,
            "mi_s": 4.0,
            "post_mi_hold_s": 1.0,
            "rest_s": 4.0,
            "transition_s": 3.0,
        },
        "phase4": {
            "window_mode": "fixed",
            "win_sec": 2.0,
            "hop_ms": 100.0,
        },
        "split": {
            "settle_s": 15.0,
        },
        "ft_defaults": {
            "use_replay": True,
            "replay_ratio": 0.10,
        },
        "overwrite_session_id": False,
    },
    "storage": {
        "save_root": "experiment_game/data/sessions",
        "save_layout": "phase_folders",
        "save_eeg": True,
        "save_events": True,
        "save_session_meta": True,
        "save_continuous_master": True,
        "save_phase_slices": True,
        "save_trial_index": True,
        "auto_phase4": False,
        "extra_copy_dir": None,
    },
    "ui": {
        "remember_last_config": True,
        "skip_setup_if_unchanged": False,
        "operator_hotkeys": True,
        "subject_feedback_mode": "none",
    },
    "extensions": {},
}


def default_run_config() -> Dict[str, Any]:
    return deepcopy(DEFAULT_RUN_CONFIG)


def merge_run_config(partial: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """浅合并分组；未知顶层键进 extensions（前向兼容）。"""
    base = default_run_config()
    if not partial:
        return base
    known = {"schema_version", "subject", "acquisition", "experiment", "storage", "ui", "extensions"}
    for key, value in partial.items():
        if key == "schema_version":
            base["schema_version"] = value
        elif key in ("subject", "acquisition", "experiment", "storage", "ui") and isinstance(
            value, dict
        ):
            if key == "acquisition" and isinstance(value.get("filter"), dict):
                filt = dict(base["acquisition"].get("filter") or {})
                filt.update(value["filter"])
                merged_acq = dict(value)
                merged_acq["filter"] = filt
                base["acquisition"].update(merged_acq)
            else:
                base[key].update(value)
        elif key == "extensions" and isinstance(value, dict):
            base["extensions"].update(value)
        elif key not in known:
            base["extensions"][key] = value
    return base


def validate_run_config(
    cfg: Dict[str, Any],
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    返回 (规范化配置, 错误列表)。错误非空时不应开诱导页/编排。
    """
    cfg = merge_run_config(cfg)
    errors: List[str] = []

    sub = cfg["subject"]
    sid = str(sub.get("subject_id") or "").strip()
    sess = str(sub.get("session_id") or "").strip()
    if not sid or not _ID_RE.match(sid):
        errors.append("subject_id 须为非空字母数字下划线")
    if not sess or not _ID_RE.match(sess):
        errors.append("session_id 须为非空字母数字下划线")
    sub["subject_id"] = sid
    sub["session_id"] = sess

    acq = cfg["acquisition"]
    exp = cfg["experiment"]
    mode = str(acq.get("board_mode") or "synthetic").lower()
    phase_mode = str(exp.get("phase_mode") or "phase2_full")
    allowed_board = ("synthetic", "cyton", "bci2a_replay")
    if mode not in allowed_board:
        errors.append("board_mode 须为 synthetic、cyton 或 bci2a_replay")
        mode = "synthetic"
    acq["board_mode"] = mode
    acq["enabled"] = bool(acq.get("enabled", True))

    port = str(acq.get("serial_port") or "").strip()
    acq["serial_port"] = port
    if acq["enabled"] and mode == "cyton" and not port:
        errors.append("真机模式须填写 serial_port（如 COM5）")

    labels = acq.get("channel_labels") or list(DEFAULT_CHANNEL_LABELS)
    if not isinstance(labels, list) or len(labels) != 8:
        errors.append("channel_labels 须为 8 个通道名")
    else:
        got = [str(x).upper() for x in labels]
        canon = [str(x).upper() for x in DEVICE_CHANNEL_LABELS]
        if sorted(got) != sorted(canon):
            errors.append(
                "channel_labels 须为设备 8 通道: " + ",".join(DEVICE_CHANNEL_LABELS)
            )
        elif got != canon:
            errors.append(
                "channel_labels 顺序须与设备序一致: " + ",".join(DEVICE_CHANNEL_LABELS)
            )
        acq["channel_labels"] = list(DEVICE_CHANNEL_LABELS)

    filt = acq.get("filter") if isinstance(acq.get("filter"), dict) else {}
    acq["filter"] = {
        "enabled": bool(filt.get("enabled", False)),
        "bandpass_low_hz": float(filt.get("bandpass_low_hz", 0.5)),
        "bandpass_high_hz": float(filt.get("bandpass_high_hz", 45.0)),
        "notch_enabled": bool(filt.get("notch_enabled", False)),
        "notch_low_hz": float(filt.get("notch_low_hz", 49.0)),
        "notch_high_hz": float(filt.get("notch_high_hz", 51.0)),
    }
    if acq["filter"]["bandpass_low_hz"] >= acq["filter"]["bandpass_high_hz"]:
        errors.append("带通低频须小于高频")
    if acq["filter"]["notch_enabled"] and (
        acq["filter"]["notch_low_hz"] >= acq["filter"]["notch_high_hz"]
    ):
        errors.append("陷波低频须小于高频")

    storage = cfg["storage"]
    storage["save_events"] = bool(storage.get("save_events", True))
    storage["save_session_meta"] = bool(storage.get("save_session_meta", True))
    if not storage["save_events"]:
        errors.append("save_events 必须开启")
    if not storage["save_session_meta"]:
        errors.append("save_session_meta 必须开启")

    # 采集开则强制写 eeg；关则强制不写
    if acq["enabled"]:
        storage["save_eeg"] = True
    else:
        storage["save_eeg"] = False

    save_root_raw = str(storage.get("save_root") or "").strip()
    if not save_root_raw:
        errors.append("save_root 不能为空")
    else:
        root = Path(save_root_raw)
        if not root.is_absolute() and repo_root is not None:
            root = (repo_root / root).resolve()
        else:
            root = root.expanduser().resolve()
        storage["save_root"] = str(root)
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"save_root 不可写: {root} ({exc})")

    try:
        exp["acquire_trials"] = int(exp.get("acquire_trials", 40))
        if exp["acquire_trials"] < 1:
            errors.append("acquire_trials 至少为 1")
    except (TypeError, ValueError):
        errors.append("acquire_trials 须为整数")
    try:
        exp["learn_trials_per_step"] = int(exp.get("learn_trials_per_step", 2))
        if exp["learn_trials_per_step"] < 1:
            errors.append("learn_trials_per_step 至少为 1")
    except (TypeError, ValueError):
        errors.append("learn_trials_per_step 须为整数")

    seed = exp.get("seed", None)
    if seed is None or seed == "":
        exp["seed"] = None
    else:
        try:
            exp["seed"] = int(seed)
        except (TypeError, ValueError):
            errors.append("seed 须为空或整数")

    exp["open_subject_page"] = bool(exp.get("open_subject_page", True))
    exp["skip_adapt"] = bool(exp.get("skip_adapt", False))
    exp["skip_learn"] = bool(exp.get("skip_learn", False))
    exp["skip_gate"] = bool(exp.get("skip_gate", False))
    exp["protocol_locked"] = bool(exp.get("protocol_locked", True))
    exp["overwrite_session_id"] = bool(exp.get("overwrite_session_id", False))
    exp["skip_v2_guidance"] = bool(exp.get("skip_v2_guidance", False))
    exp["skip_v2_calibration"] = bool(exp.get("skip_v2_calibration", False))
    exp["skip_v2_gate"] = bool(exp.get("skip_v2_gate", False))
    exp["skip_v2_game"] = bool(exp.get("skip_v2_game", False))
    ov = exp.get("v2_overrides")
    exp["v2_overrides"] = ov if isinstance(ov, dict) else {}
    v3ov = exp.get("v3_overrides")
    exp["v3_overrides"] = v3ov if isinstance(v3ov, dict) else {}
    phase_mode = str(exp.get("phase_mode") or "phase2_full")
    exp["phase_mode"] = phase_mode
    if phase_mode == "v2_session":
        try:
            from experiment_game.experiment.v2_config import V2Config

            v2_path = exp.get("v2_config_path")
            v2_cfg = V2Config.load_yaml(v2_path) if v2_path else V2Config.load_yaml()
            v2_cfg.apply_overrides(
                exp["v2_overrides"], protocol_locked=exp["protocol_locked"]
            )
            errors.extend(v2_cfg.verify_errors())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"v2 配置加载失败: {exc}")
    elif phase_mode == "v3_session":
        try:
            from experiment_game.experiment.v3_config import V3Config

            v3_path = exp.get("v3_config_path")
            v3_cfg = V3Config.load_yaml(v3_path) if v3_path else V3Config.load_yaml()
            v3_cfg.apply_overrides(
                exp["v3_overrides"], protocol_locked=exp["protocol_locked"]
            )
            errors.extend(v3_cfg.verify_errors())
            if not acq["enabled"]:
                errors.append("v3 探针会话必须开启采集（无演练模式）")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"v3 配置加载失败: {exc}")
    elif phase_mode == "sim_v3_session":
        try:
            from experiment_game.experiment.v3_config import V3Config

            v3_path = exp.get("v3_config_path")
            v3_cfg = V3Config.load_yaml(v3_path) if v3_path else V3Config.load_yaml()
            v3_cfg.apply_overrides(
                exp["v3_overrides"], protocol_locked=exp["protocol_locked"]
            )
            errors.extend(v3_cfg.verify_errors())
            if mode != "bci2a_replay":
                errors.append("仿真会话 board_mode 须为 bci2a_replay")
            exp["open_subject_page"] = True
            cfg["ui"]["subject_feedback_mode"] = "arm_reach"
            sim_ext = (cfg.get("extensions") or {}).get("sim") or {}
            if not isinstance(sim_ext, dict):
                sim_ext = {}
            run_id = str(sim_ext.get("run_id") or sub.get("session_id") or "").strip().lower()
            use_campaign_queue = bool(sim_ext.get("use_campaign_queue"))
            has_campaign = bool(sim_ext.get("campaign_manifest"))
            if not run_id.startswith("run"):
                if not (use_campaign_queue and has_campaign):
                    errors.append("仿真须指定 extensions.sim.run_id（如 run3）或启用 Campaign 队列")
            n_trials = int(sim_ext.get("session_trials_total") or 36)
            if n_trials < 6 or n_trials > 48:
                errors.append("仿真 session_trials_total 须在 6–48")
            sid_up = str(sub.get("subject_id") or "").upper()
            if not sid_up.startswith("A") or len(sid_up) != 3:
                errors.append("仿真被试须为 A01–A09")
            run_id_val = run_id if run_id.startswith("run") else "run3"
            if run_id_val.startswith("run"):
                sub["session_id"] = run_id_val
            if run_id_val.startswith("run") and sid_up.startswith("A"):
                try:
                    from experiment_game.experiment.sim.bci2a_catalog import resolve_mat_path
                    from experiment_game.experiment.sim.bci2a_mat_loader import (
                        count_run_capacity,
                        load_bci2a_run,
                    )

                    rd = load_bci2a_run(resolve_mat_path(sid_up), run_id_val)
                    _, _, _, n_max = count_run_capacity(rd)
                    if n_trials > n_max:
                        errors.append(
                            f"run {run_id_val} 最多 {n_max} 试次（含 Rest/L/R），当前 {n_trials}"
                        )
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"仿真配置加载失败: {exc}")
    elif phase_mode == "v4_session":
        try:
            from experiment_game.experiment.v4_config import V4Config

            v4_path = exp.get("v4_config_path")
            v4_cfg = V4Config.load_yaml(v4_path) if v4_path else V4Config.load_yaml()
            v4ov = exp.get("v4_overrides")
            v4_cfg.apply_overrides(v4ov if isinstance(v4ov, dict) else {})
            errors.extend(v4_cfg.verify_errors())
            if not acq["enabled"]:
                errors.append("v4 质量检测必须开启采集")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"v4 配置加载失败: {exc}")
    try:
        exp["ready_timeout_s"] = float(exp.get("ready_timeout_s", 90))
    except (TypeError, ValueError):
        exp["ready_timeout_s"] = 90.0

    # 试次时序：校验原值，再规范化为补全默认的 timing dict
    errors.extend(validate_timing_dict(exp.get("timing")))
    timing = timing_from_dict(exp.get("timing"))
    exp["timing"] = timing.to_dict()

    # Phase4 切窗参数（自动切窗与手动按钮共用默认）
    p4 = exp.get("phase4") if isinstance(exp.get("phase4"), dict) else {}
    mode = str(p4.get("window_mode") or "fixed").lower()
    if mode not in ("fixed", "slide"):
        errors.append("phase4.window_mode 须为 fixed 或 slide")
        mode = "fixed"
    try:
        p4_win = float(p4.get("win_sec", 2.0))
    except (TypeError, ValueError):
        p4_win = 2.0
        errors.append("phase4.win_sec 须为数字")
    try:
        p4_hop = float(p4.get("hop_ms", 100.0))
    except (TypeError, ValueError):
        p4_hop = 100.0
        errors.append("phase4.hop_ms 须为数字")
    if p4_win < 0.5 or p4_win > 10.0:
        errors.append("phase4.win_sec 须在 0.5–10s")
    if p4_hop < 20 or p4_hop > 2000:
        errors.append("phase4.hop_ms 须在 20–2000ms")
    # 窗必须完整落在 MI 与静息阶段内（防止把保持/过渡切进训练窗）
    if mode == "slide" and p4_win > min(timing.mi_s, timing.rest_s):
        errors.append(
            f"滑窗窗长 {p4_win:g}s 超过 MI({timing.mi_s:g}s)/静息({timing.rest_s:g}s) 阶段时长"
        )
    exp["phase4"] = {
        "window_mode": mode,
        "win_sec": p4_win,
        "hop_ms": p4_hop,
    }

    # 换场继续段开场静坐缓冲
    split = exp.get("split") if isinstance(exp.get("split"), dict) else {}
    try:
        settle_s = float(split.get("settle_s", 15.0))
    except (TypeError, ValueError):
        settle_s = 15.0
        errors.append("split.settle_s 须为数字")
    if settle_s < 5 or settle_s > 120:
        errors.append("split.settle_s 须在 5–120s")
        settle_s = 15.0
    exp["split"] = {"settle_s": settle_s}

    ui = cfg["ui"]
    ui["remember_last_config"] = bool(ui.get("remember_last_config", True))
    ui["skip_setup_if_unchanged"] = bool(ui.get("skip_setup_if_unchanged", False))
    ui["operator_hotkeys"] = bool(ui.get("operator_hotkeys", True))

    layout = str(storage.get("save_layout") or "phase_folders")
    if layout not in ("flat", "phase_folders"):
        errors.append("save_layout 须为 flat 或 phase_folders")
        layout = "phase_folders"
    storage["save_layout"] = layout
    if layout == "phase_folders":
        storage["save_continuous_master"] = True
        storage["save_phase_slices"] = True
        storage["save_trial_index"] = True
    else:
        storage["save_continuous_master"] = bool(
            storage.get("save_continuous_master", True)
        )
        storage["save_phase_slices"] = bool(storage.get("save_phase_slices", False))
        storage["save_trial_index"] = bool(storage.get("save_trial_index", True))

    storage["auto_phase4"] = bool(storage.get("auto_phase4", False))

    return cfg, errors
