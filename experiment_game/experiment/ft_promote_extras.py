"""晋升 FT 产物时附带 all4 members/overlay（供 subject / sim 共用）。"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def promote_all4_extras(
    src: Path,
    cur: Path,
    *,
    repo_root: Path,
) -> Dict[str, Any]:
    """若 FT 目录含 members/ 或 e1f_overlay.json，复制到 current 并重写 overlay 路径。"""
    src = Path(src).resolve()
    cur = Path(cur).resolve()
    root = Path(repo_root).resolve()
    cur.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {"copied_members": False, "overlay": None}

    members_src = src / "members"
    members_dst = cur / "members"
    if members_src.is_dir():
        staging = cur.parent / f".members_staging_{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(members_src, staging)
        if members_dst.exists():
            shutil.rmtree(members_dst, ignore_errors=True)
        os.replace(staging, members_dst)
        info["copied_members"] = True

    overlay_src = src / "e1f_overlay.json"
    if overlay_src.is_file():
        blob = json.loads(overlay_src.read_text(encoding="utf-8"))
        members = dict(blob.get("members") or {})
        rewritten: Dict[str, Dict[str, str]] = {}
        for name, entry in members.items():
            ent = dict(entry or {})
            three_local = members_dst / name / "best_three.pt"
            task_local = members_dst / name / "best_task.pt"
            if three_local.is_file():
                ent["three_ckpt"] = _rel(three_local, root)
            if task_local.is_file():
                ent["task_ckpt"] = _rel(task_local, root)
            elif name == "shallow" and (cur / "best_task.pt").is_file():
                ent["task_ckpt"] = _rel(cur / "best_task.pt", root)
            rewritten[name] = ent
        blob["members"] = rewritten
        blob["promoted_to_current"] = True
        from experiment_game.core.atomic_io import atomic_write_json

        atomic_write_json(cur / "e1f_overlay.json", blob)
        info["overlay"] = blob

    # 同步 force 告警（若有）
    warn = src / "force_promote_warning.json"
    if warn.is_file():
        shutil.copy2(warn, cur / "force_promote_warning.json")
        info["force_warning"] = True

    return info


def write_force_promote_warning(
    out_dir: Path,
    *,
    release_gate: Dict[str, Any],
    ft_scope: str,
    subject_id: str,
    reason: str = "auto_force_promote_on_gate_fail",
) -> Path:
    from datetime import datetime

    from experiment_game.core.atomic_io import atomic_write_json

    payload = {
        "schema": "force_promote_warning_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subject_id": subject_id,
        "ft_scope": ft_scope,
        "gate_pass": bool((release_gate or {}).get("pass")),
        "force_promoted": True,
        "reason": reason,
        "release_gate": release_gate or {},
        "message": (
            "发布门控 FAIL，已按策略强制晋升 current；请人工复核 "
            "release_gate / report。"
        ),
    }
    path = Path(out_dir) / "force_promote_warning.json"
    atomic_write_json(path, payload)
    return path
