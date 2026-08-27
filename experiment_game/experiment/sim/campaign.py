"""仿真实验 Campaign manifest（run 队列 · 去重）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_game.experiment.sim.sim_registry import (
    sim_records_campaigns,
    validate_sim_subject_id,
)


def _parse_run_num(run_id: str) -> str:
    return str(run_id).strip().lower()


def create_campaign(
    subject_id: str,
    session_queue: List[str],
    *,
    session_trials_total: int = 36,
    replay_align: str = "schedule_align",
    replay_speed: float = 4.0,
    leave_next_mode: bool = True,
    campaign_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    sid = validate_sim_subject_id(subject_id)
    queue = [_parse_run_num(r) for r in session_queue]
    if len(queue) != len(set(queue)):
        raise ValueError("session_queue 内 run 不可重复")
    if not queue:
        raise ValueError("至少 1 场 run 入队")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cid = campaign_id or f"sim_{stamp}"
    camp_dir = sim_records_campaigns(sid, repo_root=repo_root) / cid
    camp_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "campaign_id": cid,
        "subject_id": sid,
        "phase_mode": "sim_v3_session",
        "session_queue": queue,
        "session_trials_total": int(session_trials_total),
        "runs_consumed": [],
        "current_index": 0,
        "replay_align": replay_align,
        "replay_speed": float(replay_speed),
        "leave_next_mode": bool(leave_next_mode),
        "protocol": "openbmi_align_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "active",
    }
    path = camp_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(path)
    return manifest


def load_campaign(manifest_path: Path | str) -> Dict[str, Any]:
    p = Path(manifest_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["manifest_path"] = str(p.resolve())
    return data


def save_campaign(manifest: Dict[str, Any]) -> None:
    p = Path(manifest.get("manifest_path") or "")
    if not p.is_file():
        raise FileNotFoundError("manifest_path 无效")
    manifest = dict(manifest)
    manifest.pop("manifest_path", None)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pop_next_run(manifest: Dict[str, Any]) -> Optional[str]:
    """取出下一场 run；更新 manifest。"""
    idx = int(manifest.get("current_index") or 0)
    queue = list(manifest.get("session_queue") or [])
    consumed = set(manifest.get("runs_consumed") or [])
    while idx < len(queue):
        run_id = _parse_run_num(queue[idx])
        idx += 1
        manifest["current_index"] = idx
        if run_id in consumed:
            continue
        consumed.add(run_id)
        manifest["runs_consumed"] = sorted(consumed)
        save_campaign(manifest)
        return run_id
    manifest["status"] = "completed"
    save_campaign(manifest)
    return None


def campaign_summary_path(manifest: Dict[str, Any]) -> Path:
    return Path(manifest["manifest_path"]).parent / "summary.md"
