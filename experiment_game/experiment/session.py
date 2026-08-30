"""会话目录与 session.meta.json。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_game.core.channel_layout import DEFAULT_CHANNEL_LABELS
from experiment_game.core.atomic_io import atomic_write_json, atomic_write_text


@dataclass
class SessionPaths:
    root: Path
    eeg_csv: Path
    events_jsonl: Path
    meta_json: Path


@dataclass
class SessionMeta:
    subject_id: str
    session_id: str
    phase_mode: str = "acquire"
    sample_rate_hz: int = 250
    channel_labels: List[str] = field(default_factory=lambda: list(DEFAULT_CHANNEL_LABELS))
    use_synthetic: bool = True
    object: str = "cup"
    scene: str = "home_desk"
    trial_count: int = 0
    created_at: str = ""
    eeg_csv: str = "eeg.csv"
    events_jsonl: str = "events.jsonl"
    notes: str = "phase1_no_graphics"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_session_dir(
    base_dir: Path,
    subject_id: str,
    session_id: str,
    *,
    module_prefix: str = "",
) -> SessionPaths:
    """会话目录：``[module_]subject_id_session_id_YYYYmmdd_HHMMSS``。

    ``module_prefix`` 为模块前缀（如 "v3"→"v3_fnz_w01_..."），
    用于按模块隔离采集数据（需求 2026-08-30 二.1）。
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{module_prefix}_" if module_prefix else ""
    name = f"{prefix}{subject_id}_{session_id}_{stamp}"
    root = Path(base_dir) / name
    root.mkdir(parents=True, exist_ok=False)
    return SessionPaths(
        root=root,
        eeg_csv=root / "eeg.csv",
        events_jsonl=root / "events.jsonl",
        meta_json=root / "session.meta.json",
    )


def write_session_meta(path: Path, meta: SessionMeta) -> None:
    if not meta.created_at:
        meta.created_at = datetime.now().isoformat(timespec="seconds")
    atomic_write_text(
        path,
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2) + "\n",
    )


def update_session_meta(path: Path, **patch: Any) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(patch)
    atomic_write_json(path, data)
