"""操作台可选模型权重预设（底座 + 本被试 current）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_MODELS = _REPO / "experiment_game" / "data" / "models"
_SUBJECTS = _REPO / "experiment_game" / "data" / "subjects"
_SIM_SUBJECTS = _REPO / "experiment_game" / "data" / "sim_subjects"

OPENBMI_TASK = (
    "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260822_094942/task/fold0/best_task.pt"
)
OPENBMI_THREE = (
    "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260822_094942/three/fold0/best_three.pt"
)


def _exists(rel: str) -> bool:
    return (_REPO / rel).is_file()


def _meta_lineage_label(blob: Dict[str, Any], sid: str) -> str:
    return _weight_preset_label(blob, f"{sid} · current")


def _session_run_tag(session_name: str) -> str:
    for part in str(session_name or "").split("_"):
        low = part.lower()
        if low.startswith("run") and len(low) > 3 and low[3:].isdigit():
            return low
    return ""


def _read_session_lineage(session_dir_name: str, rel_prefix: str) -> Dict[str, Any]:
    meta_path = _REPO / rel_prefix / "sessions" / session_dir_name / "session.meta.json"
    if not meta_path.is_file():
        return {}
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "campaign_id": m.get("campaign_id"),
            "source_run": m.get("source_run") or m.get("session_id"),
            "session_id": m.get("session_id"),
        }
    except Exception:
        return {}


def _enrich_weight_meta(blob: Dict[str, Any], rel_prefix: str) -> Dict[str, Any]:
    out = dict(blob)
    if out.get("campaign_id") and out.get("source_run"):
        return out
    ses = out.get("session")
    if not ses and out.get("sessions"):
        ses_list = out.get("sessions")
        if isinstance(ses_list, list) and ses_list:
            ses = ses_list[0]
    if not ses:
        return out
    first = str(ses).split(" + ")[0].strip()
    line = _read_session_lineage(first, rel_prefix)
    if line.get("campaign_id"):
        out.setdefault("campaign_id", line["campaign_id"])
    if line.get("source_run"):
        out.setdefault("source_run", line["source_run"])
    if line.get("session_id"):
        out.setdefault("session_id", line["session_id"])
    return out


def _weight_preset_label(meta: Dict[str, Any], head: str) -> str:
    gate = (meta.get("release_gate") or {}).get("pass")
    gate_s = "PASS" if gate is True else ("FAIL" if gate is False else "—")
    campaign = meta.get("campaign_id")
    source_run = meta.get("source_run") or _session_run_tag(
        str(meta.get("session") or meta.get("sessions") or "")
    )
    parts = [head]
    if campaign:
        parts.append(str(campaign))
    if source_run:
        parts.append(str(source_run))
    parts.append(f"gate {gate_s}")
    return " · ".join(parts)


def _ft_run_label(meta: Dict[str, Any], run_stamp: str, subject_key: str) -> str:
    return _weight_preset_label(meta, f"{subject_key} · FT")


def _append_ft_run_presets(
    presets: List[Dict[str, Any]],
    *,
    subject_key: str,
    rel_prefix: str,
    max_runs: int = 8,
    skip_paths: Optional[set[str]] = None,
) -> None:
    ft_root = _REPO / rel_prefix / "models" / "ft_runs"
    if not ft_root.is_dir():
        return
    skip = skip_paths or set()
    dirs = sorted(
        [p for p in ft_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in dirs[:max_runs]:
        task_p = d / "best_task.pt"
        three_p = d / "best_three.pt"
        if not task_p.is_file() or not three_p.is_file():
            continue
        task = f"{rel_prefix}/models/ft_runs/{d.name}/best_task.pt"
        three = f"{rel_prefix}/models/ft_runs/{d.name}/best_three.pt"
        norm = f"{task}|{three}"
        if norm in skip:
            continue
        label = f"FT {d.name}"
        meta_blob: Dict[str, Any] = {}
        meta = d / "meta.json"
        if meta.is_file():
            try:
                meta_blob = _enrich_weight_meta(
                    json.loads(meta.read_text(encoding="utf-8")),
                    rel_prefix,
                )
                label = _ft_run_label(meta_blob, d.name, subject_key)
            except Exception:
                pass
        gate_pass = (meta_blob.get("release_gate") or {}).get("pass")
        presets.append(
            {
                "id": f"ft_{d.name}",
                "label": label,
                "task": task,
                "three": three,
                "ok": True,
                "subject_id": subject_key,
                "kind": "ft_run",
                "ft_stamp": d.name,
                "campaign_id": meta_blob.get("campaign_id"),
                "source_run": meta_blob.get("source_run"),
                "release_pass": gate_pass,
            }
        )


def _append_current_preset(
    presets: List[Dict[str, Any]],
    *,
    subject_key: str,
    rel_prefix: str,
    preset_id: str,
) -> None:
    cur = _REPO / rel_prefix / "models" / "current"
    task_p = cur / "best_task.pt"
    three_p = cur / "best_three.pt"
    if not task_p.is_file() or not three_p.is_file():
        return
    task = f"{rel_prefix}/models/current/best_task.pt"
    three = f"{rel_prefix}/models/current/best_three.pt"
    label = f"{subject_key} current"
    meta_blob: Dict[str, Any] = {}
    meta = cur / "meta.json"
    if meta.is_file():
        try:
            meta_blob = _enrich_weight_meta(
                json.loads(meta.read_text(encoding="utf-8")),
                rel_prefix,
            )
            label = _meta_lineage_label(meta_blob, subject_key)
        except Exception:
            pass
    gate_pass = (meta_blob.get("release_gate") or {}).get("pass")
    presets.append(
        {
            "id": preset_id,
            "label": label,
            "task": task,
            "three": three,
            "ok": True,
            "subject_id": subject_key,
            "kind": "current",
            "campaign_id": meta_blob.get("campaign_id"),
            "source_run": meta_blob.get("source_run"),
            "release_pass": gate_pass,
        }
    )


def list_model_presets(
    *,
    subject_id: Optional[str] = None,
    sim_mode: bool = False,
) -> List[Dict[str, Any]]:
    """返回 [{id, label, task, three, ok}]；登录后仅含 OpenBMI + 本被试权重。"""
    presets: List[Dict[str, Any]] = [
        {
            "id": "openbmi_baseline",
            "label": "OpenBMI 底座（零样本 · run_20260822_094942）",
            "task": OPENBMI_TASK,
            "three": OPENBMI_THREE,
            "ok": _exists(OPENBMI_TASK) and _exists(OPENBMI_THREE),
            "subject_id": None,
        }
    ]

    sid = (subject_id or "").strip()
    sid_lower = sid.lower()
    sid_upper = sid.upper()

    def _allowed(name: str) -> bool:
        if not sid:
            return subject_id is None
        return name == sid_lower or name == sid_upper

    roots: List[tuple[str, Path, str]] = []
    if _SUBJECTS.is_dir():
        for d in sorted(p for p in _SUBJECTS.iterdir() if p.is_dir()):
            if sid and not _allowed(d.name):
                continue
            roots.append((d.name, d, f"experiment_game/data/subjects/{d.name}"))
    if sim_mode or (sid_upper.startswith("A0") and len(sid_upper) == 3):
        if _SIM_SUBJECTS.is_dir():
            for d in sorted(p for p in _SIM_SUBJECTS.iterdir() if p.is_dir()):
                if sid and not _allowed(d.name):
                    continue
                roots.append((d.name, d, f"experiment_game/data/sim_subjects/{d.name}"))

    seen: set[str] = set()
    for name, _d, rel_prefix in roots:
        pid = f"subject_{name}"
        if pid in seen:
            continue
        seen.add(pid)
        _append_current_preset(presets, subject_key=name, rel_prefix=rel_prefix, preset_id=pid)
        cur_paths: set[str] = set()
        for p in presets:
            if p.get("id") == pid:
                cur_paths.add(f"{p['task']}|{p['three']}")
        _append_ft_run_presets(
            presets,
            subject_key=name,
            rel_prefix=rel_prefix,
            skip_paths=cur_paths,
        )

    if not sid and _MODELS.is_dir():
        for d in sorted(p for p in _MODELS.iterdir() if p.is_dir()):
            task = f"experiment_game/data/models/{d.name}/best_task.pt"
            three = f"experiment_game/data/models/{d.name}/best_three.pt"
            if not (_exists(task) and _exists(three)):
                continue
            meta_label = d.name
            meta = d / "meta.json"
            if meta.is_file():
                try:
                    blob = json.loads(meta.read_text(encoding="utf-8"))
                    meta_label = _meta_lineage_label(blob, blob.get("subject_id") or d.name)
                except Exception:
                    pass
            presets.append(
                {
                    "id": d.name,
                    "label": meta_label,
                    "task": task,
                    "three": three,
                    "ok": True,
                    "subject_id": d.name,
                }
            )
    return presets


def match_preset_id(
    task: str,
    three: str,
    *,
    subject_id: Optional[str] = None,
    sim_mode: bool = False,
) -> str:
    t = (task or "").replace("\\", "/")
    th = (three or "").replace("\\", "/")
    for p in list_model_presets(subject_id=subject_id, sim_mode=sim_mode):
        if p["task"].replace("\\", "/") == t and p["three"].replace("\\", "/") == th:
            return str(p["id"])
    return "custom"


def latest_preset_for_campaign(
    campaign_id: str,
    *,
    subject_id: Optional[str] = None,
    sim_mode: bool = False,
    prefer_pass: bool = True,
) -> Optional[Dict[str, Any]]:
    """当前 Campaign 内最新 FT 权重（按 ft_stamp 时间倒序）。"""
    cid = str(campaign_id or "").strip()
    if not cid:
        return None
    hits = [
        p
        for p in list_model_presets(subject_id=subject_id, sim_mode=sim_mode)
        if p.get("kind") == "ft_run" and p.get("campaign_id") == cid and p.get("ok")
    ]
    if not hits:
        return None
    if prefer_pass:
        passed = [p for p in hits if p.get("release_pass") is True]
        if passed:
            hits = passed
    return hits[0]


def active_weights_from_yaml() -> Dict[str, Any]:
    """读当前 v3（优先）yaml 中的权重路径。"""
    from experiment_game.experiment.v3_config import V3Config

    cfg = V3Config.load_yaml()
    task = cfg.s3_task_ckpt
    three = cfg.s3_three_ckpt
    return {
        "task": task,
        "three": three,
        "preset_id": match_preset_id(task, three),
        "task_ok": _exists(task),
        "three_ok": _exists(three),
    }


def short_weight_label(rel: str) -> str:
    """操作台状态栏短显示。"""
    p = (rel or "").replace("\\", "/")
    if not p:
        return "—"
    parts = p.split("/")
    if "models" in parts:
        i = parts.index("models")
        if i + 1 < len(parts):
            return parts[i + 1]
    if "openbmi" in p.lower() or "5070_baseline" in p:
        return "openbmi_baseline"
    return parts[-2] if len(parts) >= 2 else parts[-1]
