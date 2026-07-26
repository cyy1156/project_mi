"""会话目录与 session.meta.json。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_game.acquisition.service import DEFAULT_CHANNEL_LABELS


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
) -> SessionPaths:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{subject_id}_{session_id}_{stamp}"
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
    path.write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_session_meta(path: Path, **patch: Any) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(patch)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
