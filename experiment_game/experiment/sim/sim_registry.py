"""仿真被试 A01–A09 工作区（data/sim_subjects/）。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_game.experiment.sim.bci2a_catalog import resolve_mat_path

_REPO = Path(__file__).resolve().parents[3]
_SIM_ROOT = _REPO / "experiment_game" / "data" / "sim_subjects"

_SIM_ID_RE = re.compile(r"^A0[1-9]$")


def validate_sim_subject_id(subject_id: str) -> str:
    sid = str(subject_id or "").strip().upper()
    if not _SIM_ID_RE.match(sid):
        raise ValueError("仿真被试须为 A01–A09（大写）")
    return sid


def sim_subjects_root() -> Path:
    return _SIM_ROOT


def sim_subject_root(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    sid = validate_sim_subject_id(subject_id)
    root = Path(repo_root or _REPO)
    return root / "experiment_game" / "data" / "sim_subjects" / sid


def sim_sessions_dir(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    return sim_subject_root(subject_id, repo_root=repo_root) / "sessions"


def sim_models_current(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    return sim_subject_root(subject_id, repo_root=repo_root) / "models" / "current"


def sim_records_campaigns(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    return sim_subject_root(subject_id, repo_root=repo_root) / "records" / "campaigns"


def rel_repo_path(path: Path, *, repo_root: Optional[Path] = None) -> str:
    root = Path(repo_root or _REPO).resolve()
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sim_models_ft_runs(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    return sim_subject_root(subject_id, repo_root=repo_root) / "models" / "ft_runs"


def list_sim_sessions(subject_id: str, *, repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    from experiment_game.experiment.subject_registry import _session_metrics

    sid = validate_sim_subject_id(subject_id)
    base = sim_sessions_dir(sid, repo_root=repo_root)
    if not base.is_dir():
        return []
    dirs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(f"{sid}_"))
    return [_session_metrics(d) for d in dirs]


def current_sim_model_paths(subject_id: str, *, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    sid = validate_sim_subject_id(subject_id)
    root = Path(repo_root or _REPO)
    cur = sim_models_current(sid, repo_root=root)
    task = cur / "best_task.pt"
    three = cur / "best_three.pt"
    ok = task.is_file() and three.is_file()
    meta: Dict[str, Any] = {}
    meta_path = cur / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return {
        "path": rel_repo_path(cur, repo_root=root) if ok else None,
        "task_ckpt": rel_repo_path(task, repo_root=root) if task.is_file() else None,
        "three_ckpt": rel_repo_path(three, repo_root=root) if three.is_file() else None,
        "ok": ok,
        "release_pass": (meta.get("release_gate") or {}).get("pass"),
        "updated_at": meta.get("session") or meta.get("sessions"),
        "lineage_session_ids": meta.get("session_ids") or meta.get("session"),
    }


def suggest_next_run(subject_id: str, *, repo_root: Optional[Path] = None) -> str:
    """建议下一场 run（未在 sessions 目录出现过的最小 runN）。"""
    from experiment_game.experiment.sim.bci2a_catalog import list_subject_runs

    sid = validate_sim_subject_id(subject_id)
    used: set[str] = set()
    for s in list_sim_sessions(sid, repo_root=repo_root):
        if s.get("record_excluded"):
            continue
        sid_part = str(s.get("session_id") or "")
        if sid_part.startswith("run"):
            used.add(sid_part.lower())
    for r in list_subject_runs(sid):
        rid = str(r.get("run_id") or "").lower()
        if rid and rid not in used:
            return rid
    runs = list_subject_runs(sid)
    if runs:
        return str(runs[0].get("run_id") or "run3")
    return "run3"


def build_sim_index(subject_id: str, *, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    sid = validate_sim_subject_id(subject_id)
    sessions = list_sim_sessions(sid, repo_root=repo_root)
    current = current_sim_model_paths(sid, repo_root=repo_root)
    nxt = suggest_next_run(sid, repo_root=repo_root)
    idx = {
        "subject_id": sid,
        "sim_mode": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "sessions": sessions,
        "session_count_official": len([s for s in sessions if not s.get("record_excluded")]),
        "current_model": current,
        "suggest_session_id": nxt,
    }
    index_path = sim_subject_root(sid, repo_root=repo_root) / "index.json"
    index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sub_json = sim_subject_root(sid, repo_root=repo_root) / "subject.json"
    if sub_json.is_file():
        data = json.loads(sub_json.read_text(encoding="utf-8"))
        data["session_count"] = len([s for s in sessions if not s.get("record_excluded")])
        data["next_session_suggest"] = nxt
        sub_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return idx


def new_sim_ft_run_dir(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    sid = validate_sim_subject_id(subject_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = sim_models_ft_runs(sid, repo_root=repo_root) / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def promote_sim_ft_to_current(
    subject_id: str,
    ft_run_dir: Path,
    *,
    repo_root: Optional[Path] = None,
    reason: str = "",
) -> Dict[str, Any]:
    sid = validate_sim_subject_id(subject_id)
    root = Path(repo_root or _REPO)
    src = Path(ft_run_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"FT 目录不存在: {src}")
    for name in ("best_task.pt", "best_three.pt"):
        if not (src / name).is_file():
            raise FileNotFoundError(f"缺少 {name} in {src}")
    cur = sim_models_current(sid, repo_root=root)
    from experiment_game.core.atomic_io import atomic_copy_files_into, atomic_write_json

    atomic_copy_files_into(
        src,
        cur,
        ("best_task.pt", "best_three.pt", "meta.json", "report.md", "release_gate.json"),
    )
    promote_log = {
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "from_ft_run": rel_repo_path(src, repo_root=root),
        "reason": reason or "operator_confirmed",
    }
    atomic_write_json(cur / "promote_log.json", promote_log)
    build_sim_index(sid, repo_root=root)
    return {
        "ok": True,
        "current_dir": rel_repo_path(cur, repo_root=root),
        "weights": current_sim_model_paths(sid, repo_root=root),
        "promote_log": promote_log,
    }


def list_campaigns(subject_id: str, *, repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    sid = validate_sim_subject_id(subject_id)
    base = sim_records_campaigns(sid, repo_root=repo_root)
    out: List[Dict[str, Any]] = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        mf = d / "manifest.json"
        if not mf.is_file():
            continue
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            data["manifest_path"] = str(mf.resolve())
            from experiment_game.experiment.model_presets import campaign_locked_model_preset_id

            rem = [
                r
                for r in (data.get("session_queue") or [])
                if r not in set(data.get("runs_consumed") or [])
            ]
            data["remaining_runs"] = rem
            data["remaining_count"] = len(rem)
            locked = campaign_locked_model_preset_id(
                str(data.get("campaign_id") or ""),
                subject_id=sid,
                sim_mode=True,
                manifest=data,
            )
            if locked:
                data["locked_model_preset_id"] = locked
            out.append(data)
        except Exception:
            continue
    return out


def login_sim_subject(
    subject_id: str,
    *,
    repo_root: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    sid = validate_sim_subject_id(subject_id)
    root = sim_subject_root(sid, repo_root=repo_root)
    for sub in (
        root,
        sim_sessions_dir(sid, repo_root=repo_root),
        sim_models_current(sid, repo_root=repo_root),
        root / "models" / "ft_runs",
        root / "analysis",
        sim_records_campaigns(sid, repo_root=repo_root),
    ):
        sub.mkdir(parents=True, exist_ok=True)

    mat_path = resolve_mat_path(sid, data_dir=data_dir)
    sub_json = root / "subject.json"
    now = datetime.now().isoformat(timespec="seconds")
    if sub_json.is_file():
        data = json.loads(sub_json.read_text(encoding="utf-8"))
        data["last_login_at"] = now
    else:
        data = {
            "subject_id": sid,
            "sim_mode": True,
            "source_mat": str(mat_path),
            "created_at": now,
            "last_login_at": now,
        }
    sub_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from experiment_game.experiment.sim.bci2a_catalog import list_subject_runs

    index = build_sim_index(sid, repo_root=repo_root)
    return {
        "subject_id": sid,
        "sim_mode": True,
        "subject_root": rel_repo_path(root, repo_root=repo_root),
        "sessions_dir": rel_repo_path(sim_sessions_dir(sid, repo_root=repo_root), repo_root=repo_root),
        "source_mat": str(mat_path),
        "runs": list_subject_runs(sid, data_dir=data_dir),
        "campaigns": list_campaigns(sid, repo_root=repo_root),
        "subject": data,
        "index": index,
        "sessions": index.get("sessions") or [],
        "suggest_session_id": index.get("suggest_session_id"),
        "current_weights": current_sim_model_paths(sid, repo_root=repo_root),
    }


def storage_paths_for_sim(subject_id: str, *, repo_root: Optional[Path] = None) -> Dict[str, str]:
    sid = validate_sim_subject_id(subject_id)
    return {
        "save_root": rel_repo_path(sim_sessions_dir(sid, repo_root=repo_root), repo_root=repo_root),
        "subject_root": rel_repo_path(sim_subject_root(sid, repo_root=repo_root), repo_root=repo_root),
    }
