"""操作台可选模型权重预设（底座 + 本被试 current）。"""

from __future__ import annotations

import json
import re
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

E1F_CONFIG = "experiment_game/config/e1f_four_member.json"
E1F_TASK = (
    "code/train_lab/out/5090_alg_incr_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260823_095327/task/fold0/best_task.pt"
)
E1F_THREE = (
    "code/train_lab/out/5090_alg_incr_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260823_095327/three/fold0/best_three.pt"
)


def _exists(rel: str) -> bool:
    return (_REPO / rel).is_file()


def _e1f_missing_weights() -> bool:
    """不依赖 adapt_engine 导入（操作台进程未必已加 code/ 到 sys.path）。"""
    cfg_path = _REPO / E1F_CONFIG
    if not cfg_path.is_file():
        return True
    try:
        blob = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    task = str(blob.get("task_ckpt") or "").strip()
    if task:
        p = _REPO / task if not Path(task).is_absolute() else Path(task)
        if not p.is_file():
            return True
    for m in blob.get("members") or []:
        rel = str(m.get("three_ckpt") or "").strip()
        if not rel:
            return True
        p = _REPO / rel if not Path(rel).is_absolute() else Path(rel)
        if not p.is_file():
            return True
        trel = str(m.get("task_ckpt") or "").strip()
        if trel:
            tp = _REPO / trel if not Path(trel).is_absolute() else Path(trel)
            if not tp.is_file():
                return True
    return False


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


def _session_id_tag(name: str) -> str:
    """从目录名 / session_id 抽出 ws02 / w01 / run3 等短标签。"""
    s = str(name or "").strip()
    if not s:
        return ""
    # 已是短 id：优先 ws/run/ses，其次 wNN（勿把 ws01 误拆成 w）
    if re.fullmatch(r"(?i)(ws|run|ses)\d+", s):
        return s.lower()
    if re.fullmatch(r"(?i)w\d+", s):
        return s.lower()
    m = re.search(r"(?i)(?:^|_)((?:ws|run|ses)\d+)(?:_|$)", s)
    if m:
        return m.group(1).lower()
    m = re.search(r"(?i)(?:^|_)(w\d+)(?:_|$)", s)
    if m:
        return m.group(1).lower()
    # session_lineage 项可能只有 session 目录名
    parts = s.split("_")
    for p in parts:
        if re.fullmatch(r"(?i)(ws|run|ses)\d+", p):
            return p.lower()
    for p in parts:
        if re.fullmatch(r"(?i)w\d+", p):
            return p.lower()
    return ""


def _lineage_run_span(meta: Dict[str, Any]) -> str:
    """Leave-Next / 多会话：ws02+ws03+ws04→ws05；单场仍用 source_run。"""
    train_tags: List[str] = []
    hold_tags: List[str] = []

    for key, bucket in (("train_sessions", train_tags), ("heldout_sessions", hold_tags)):
        for item in meta.get(key) or []:
            tag = _session_id_tag(item)
            if tag and tag not in bucket:
                bucket.append(tag)

    if not train_tags:
        for row in meta.get("session_lineage") or []:
            if not isinstance(row, dict):
                continue
            tag = _session_id_tag(row.get("session_id") or row.get("source_run") or row.get("session") or "")
            if tag and tag not in train_tags:
                train_tags.append(tag)

    if not train_tags:
        single = meta.get("source_run") or _session_run_tag(
            str(meta.get("session") or meta.get("sessions") or "")
        )
        if single:
            train_tags = [str(single)]

    if train_tags and hold_tags:
        return f"{'+'.join(train_tags)}→{'+'.join(hold_tags)}"
    if train_tags:
        return "+".join(train_tags)
    return ""


def _lineage_label(meta: Dict[str, Any], head: str) -> str:
    """权重谱系（不含模型架构名）。"""
    gate = (meta.get("release_gate") or {}).get("pass")
    gate_s = "PASS" if gate is True else ("FAIL" if gate is False else "—")
    campaign = meta.get("campaign_id")
    run_span = _lineage_run_span(meta)
    parts = [head]
    if campaign:
        parts.append(str(campaign))
    if run_span:
        parts.append(run_span)
    replay = str(meta.get("replay_pool") or "").lower()
    if meta.get("no_replay") or replay in ("", "none"):
        if meta.get("leave_next") or "leave_next" in str(meta.get("finetune_mode") or ""):
            parts.append("noreplay")
    elif replay and replay != "t0":
        parts.append(f"replay:{replay}")
    parts.append(f"gate {gate_s}")
    return " · ".join(parts)


def _weight_preset_label(meta: Dict[str, Any], head: str) -> str:
    return _lineage_label(meta, head)


def resolve_model_name(
    *,
    preset_id: Optional[str] = None,
    readout_mode: Optional[str] = None,
) -> str:
    """模型架构名（操作台「当前模型」）。"""
    pid = (preset_id or "").strip()
    ro = (readout_mode or "").lower()
    if ro == "e1f" or pid == "e1f_four_member":
        return "E1f 四成员（OpenBMI · test 0.6173）"
    if pid == "openbmi_baseline":
        return "OpenBMI Shallow（T0）"
    if ro in ("serial_gating", ""):
        return "OpenBMI Shallow（T0）"
    return "OpenBMI Shallow（T0）"


def compose_weight_label(model_name: str, lineage: str) -> str:
    """完整权重名（预设下拉）：模型名 · 谱系。"""
    lineage = (lineage or "").strip()
    if not lineage:
        return model_name
    if lineage.startswith(model_name):
        return lineage
    return f"{model_name} · {lineage}"


def _default_readout_mode() -> str:
    try:
        from experiment_game.experiment.v3_config import V3Config

        return str(getattr(V3Config.load_yaml(), "readout_mode", "") or "")
    except Exception:
        return ""


def _enrich_preset_names(preset: Dict[str, Any], *, default_readout: str = "") -> Dict[str, Any]:
    ro = str(preset.get("readout_mode") or default_readout or "")
    pid = str(preset.get("id") or "")
    model_name = resolve_model_name(preset_id=pid, readout_mode=ro)
    preset["model_name"] = model_name
    raw_label = str(preset.get("label") or "")
    if preset.get("kind") in ("ft_run", "current") or pid not in (
        "openbmi_baseline",
        "e1f_four_member",
    ):
        weight_label = compose_weight_label(model_name, raw_label)
    else:
        weight_label = raw_label if raw_label.startswith(model_name) else compose_weight_label(
            model_name, raw_label.split(" · ", 1)[-1] if " · " in raw_label else raw_label
        )
    preset["weight_label"] = weight_label
    preset["label"] = weight_label
    if ro:
        preset.setdefault("readout_mode", ro)
    return preset


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
        ro = str(meta_blob.get("readout_mode") or _default_readout_mode() or "")
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
                "readout_mode": ro,
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
    ro = str(meta_blob.get("readout_mode") or _default_readout_mode() or "")
    overlay_rel = f"{rel_prefix}/models/current/e1f_overlay.json"
    overlay_p = cur / "e1f_overlay.json"
    has_overlay = overlay_p.is_file()
    if has_overlay or str(meta_blob.get("ft_scope") or "").lower() == "all4":
        ro = "e1f"
    entry = {
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
        "readout_mode": ro,
    }
    if has_overlay:
        entry["e1f_overlay_path"] = overlay_rel
        entry["e1f_config_path"] = E1F_CONFIG
        entry["primary_judge_mode"] = "majority"
        entry["ft_scope"] = "all4"
    presets.append(entry)


def list_model_presets(
    *,
    subject_id: Optional[str] = None,
    sim_mode: bool = False,
) -> List[Dict[str, Any]]:
    """返回 [{id, label, task, three, ok}]；登录后仅含 OpenBMI + 本被试权重。"""
    presets: List[Dict[str, Any]] = [
        {
            "id": "openbmi_baseline",
            "label": "OpenBMI Shallow（T0） · 零样本 · run_20260822_094942",
            "task": OPENBMI_TASK,
            "three": OPENBMI_THREE,
            "ok": _exists(OPENBMI_TASK) and _exists(OPENBMI_THREE),
            "subject_id": None,
            "readout_mode": "serial_gating",
        },
        {
            "id": "e1f_four_member",
            "label": "E1f 四成员（OpenBMI · test 0.6173） · 零样本",
            "task": E1F_TASK,
            "three": E1F_THREE,
            "ok": _exists(E1F_CONFIG) and not _e1f_missing_weights(),
            "subject_id": None,
            "readout_mode": "e1f",
            "e1f_config_path": E1F_CONFIG,
            "primary_judge_mode": "majority",
        },
    ]
    default_readout = _default_readout_mode()

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
    for i, p in enumerate(presets):
        presets[i] = _enrich_preset_names(p, default_readout=default_readout)
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


def resolve_model_display_label(
    *,
    task: str = "",
    three: str = "",
    preset_id: Optional[str] = None,
    readout_mode: Optional[str] = None,
    subject_id: Optional[str] = None,
    sim_mode: bool = False,
) -> str:
    """操作台「当前模型」：仅模型架构名。"""
    _ = subject_id, sim_mode, task, three
    return resolve_model_name(preset_id=preset_id, readout_mode=readout_mode)


def resolve_weight_display_label(
    *,
    task: str = "",
    three: str = "",
    preset_id: Optional[str] = None,
    readout_mode: Optional[str] = None,
    subject_id: Optional[str] = None,
    sim_mode: bool = False,
) -> str:
    """操作台「预设」：完整权重名。"""
    pid = (preset_id or "").strip()
    if pid and pid != "custom":
        for p in list_model_presets(subject_id=subject_id, sim_mode=sim_mode):
            if p.get("id") == pid:
                return str(p.get("weight_label") or p.get("label") or pid)
    t = (task or "").replace("\\", "/")
    th = (three or "").replace("\\", "/")
    for p in list_model_presets(subject_id=subject_id, sim_mode=sim_mode):
        if p.get("task") == t and p.get("three") == th:
            return str(p.get("weight_label") or p.get("label") or p.get("id") or "—")
    model_name = resolve_model_name(preset_id=pid, readout_mode=readout_mode)
    if th or t:
        return compose_weight_label(model_name, f"自定义 · {short_weight_label(th or t)}")
    return model_name if model_name else "未配置"


def campaign_locked_model_preset_id(
    campaign_id: str,
    *,
    subject_id: Optional[str] = None,
    sim_mode: bool = False,
    manifest: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Campaign 已开跑后锁定的零样本模型 preset id。"""
    cid = str(campaign_id or "").strip()
    if not cid:
        return None
    latest = latest_preset_for_campaign(
        cid, subject_id=subject_id, sim_mode=sim_mode, prefer_pass=False
    )
    if latest:
        if str(latest.get("readout_mode") or "").lower() == "e1f":
            return "e1f_four_member"
        return "openbmi_baseline"
    mf = manifest or {}
    for item in mf.get("sessions_completed") or []:
        sdir = Path(str(item.get("session_dir") or ""))
        meta_path = sdir / "session.meta.json"
        if not meta_path.is_file():
            continue
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            _sum = m.get("v3_summary") or {}
            _eff = m.get("v3_config_effective") or _sum.get("v3_config_effective") or {}
            ro = str(
                _sum.get("readout_mode")
                or _eff.get("readout_mode")
                or m.get("readout_mode")
                or ""
            )
            if ro.lower() == "e1f":
                return "e1f_four_member"
            if ro:
                return "openbmi_baseline"
        except Exception:
            continue
    return None


def active_weights_from_yaml() -> Dict[str, Any]:
    """读当前 v3（优先）yaml 中的权重路径。"""
    from experiment_game.experiment.v3_config import V3Config

    cfg = V3Config.load_yaml()
    task = cfg.s3_task_ckpt
    three = cfg.s3_three_ckpt
    readout = getattr(cfg, "readout_mode", "") or ""
    preset_id = match_preset_id(task, three)
    if readout.lower() == "e1f":
        preset_id = "e1f_four_member"
    label = resolve_weight_display_label(
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
        "label": label,
        "weight_label": label,
        "model_label": model_label,
        "task_ok": _exists(task) if task else True,
        "three_ok": _exists(three) if three else not readout.lower() == "e1f",
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
