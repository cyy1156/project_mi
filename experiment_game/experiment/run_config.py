"""操作台 run_config schema 校验与默认值（UI-1）。"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_game.acquisition.service import DEFAULT_CHANNEL_LABELS

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
            "enabled": True,
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 45.0,
            "notch_low_hz": 49.0,
            "notch_high_hz": 51.0,
        },
        "markers_lsl": True,
    },
    "experiment": {
        "acquire_trials": 8,
        "learn_trials_per_step": 2,
        "skip_adapt": False,
        "skip_learn": False,
        "skip_gate": False,
        "seed": None,
        "open_subject_page": True,
        "ready_timeout_s": 90,
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
    mode = str(acq.get("board_mode") or "synthetic").lower()
    if mode not in ("synthetic", "cyton"):
        errors.append("board_mode 须为 synthetic 或 cyton")
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
        acq["channel_labels"] = [str(x) for x in labels]

    filt = acq.get("filter") if isinstance(acq.get("filter"), dict) else {}
    acq["filter"] = {
        "enabled": bool(filt.get("enabled", True)),
        "bandpass_low_hz": float(filt.get("bandpass_low_hz", 0.5)),
        "bandpass_high_hz": float(filt.get("bandpass_high_hz", 45.0)),
        "notch_low_hz": float(filt.get("notch_low_hz", 49.0)),
        "notch_high_hz": float(filt.get("notch_high_hz", 51.0)),
    }
    if acq["filter"]["bandpass_low_hz"] >= acq["filter"]["bandpass_high_hz"]:
        errors.append("带通低频须小于高频")
    if acq["filter"]["notch_low_hz"] >= acq["filter"]["notch_high_hz"]:
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

    exp = cfg["experiment"]
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
    try:
        exp["ready_timeout_s"] = float(exp.get("ready_timeout_s", 90))
    except (TypeError, ValueError):
        exp["ready_timeout_s"] = 90.0

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
