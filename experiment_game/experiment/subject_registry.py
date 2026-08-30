"""被试工作区：登录建档、session 索引、current 权重晋升。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_game.experiment.session_electrode import scan_session_electrodes, _find_eeg_csv

_REPO = Path(__file__).resolve().parents[2]
_SUBJECTS_ROOT = _REPO / "experiment_game" / "data" / "subjects"
_LEGACY_SESSIONS = _REPO / "experiment_game" / "data" / "sessions"
_LEGACY_MODELS = _REPO / "experiment_game" / "data" / "models"

# OpenBMI-Align：每 MI 试次约 11 个 task 窗（3.0…4.0s hop100）；Exp29 ~300 窗/run
OPENBMI_WINDOWS_PER_MI_TRIAL = 11
EXP29_WINDOWS_PER_RUN = 300

_SUBJECT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,14}$")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def repo_root() -> Path:
    return _REPO


def subjects_root() -> Path:
    return _SUBJECTS_ROOT


def validate_subject_id(subject_id: str) -> str:
    sid = str(subject_id or "").strip().lower()
    if not _SUBJECT_ID_RE.match(sid):
        raise ValueError(
            "subject_id 须为小写字母开头、仅含 a-z0-9_，长度 2–15（如 fnz、sub01）"
        )
    return sid


def subject_root(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    sid = validate_subject_id(subject_id)
    root = Path(repo_root or _REPO)
    return root / "experiment_game" / "data" / "subjects" / sid


def sessions_dir(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    return subject_root(subject_id, repo_root=repo_root) / "sessions"


def models_current_dir(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    return subject_root(subject_id, repo_root=repo_root) / "models" / "current"


def models_ft_runs_dir(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    return subject_root(subject_id, repo_root=repo_root) / "models" / "ft_runs"


def rel_repo_path(path: Path, *, repo_root: Optional[Path] = None) -> str:
    root = Path(repo_root or _REPO).resolve()
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def login_subject(
    subject_id: str,
    *,
    display_name: str = "",
    notes: str = "",
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """创建或加载被试工作区。"""
    sid = validate_subject_id(subject_id)
    root = subject_root(sid, repo_root=repo_root)
    sess = sessions_dir(sid, repo_root=repo_root)
    current = models_current_dir(sid, repo_root=repo_root)
    ft_runs = models_ft_runs_dir(sid, repo_root=repo_root)
    for p in (root, sess, current, ft_runs):
        p.mkdir(parents=True, exist_ok=True)

    sub_json = root / "subject.json"
    now = datetime.now().isoformat(timespec="seconds")
    if sub_json.is_file():
        data = json.loads(sub_json.read_text(encoding="utf-8"))
        data["last_login_at"] = now
        if display_name:
            data["display_name"] = display_name
        if notes:
            data["notes"] = notes
    else:
        data = {
            "subject_id": sid,
            "display_name": display_name or "",
            "created_at": now,
            "last_login_at": now,
            "session_count": 0,
            "next_session_suggest": "w01",
            "notes": notes or "",
        }
    sub_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index = build_index(sid, repo_root=repo_root)
    by_board = index.get("suggest_session_ids_by_board") or suggest_session_ids_by_board(
        sid, repo_root=repo_root
    )
    return {
        "subject_id": sid,
        "subject_root": rel_repo_path(root, repo_root=repo_root),
        "sessions_dir": rel_repo_path(sess, repo_root=repo_root),
        "current_model_dir": rel_repo_path(current, repo_root=repo_root),
        "subject": data,
        "index": index,
        "sessions": index.get("sessions") or [],
        "suggest_session_id": index.get("suggest_session_id")
        or suggest_session_id(sid, repo_root=repo_root),
        "suggest_session_ids_by_board": by_board,
        "current_weights": current_model_paths(sid, repo_root=repo_root),
    }


def _parse_session_name(name: str) -> Tuple[str, str, str]:
    """fnz_ws01_20260826_164149 → (subject, session_id, stamp)."""
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], "_".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return name, "ses01", ""


def _discover_session_dirs(subject_id: str, *, repo_root: Optional[Path] = None) -> List[Path]:
    sid = validate_subject_id(subject_id)
    root = Path(repo_root or _REPO)
    found: Dict[str, Path] = {}
    prefixes = (f"{sid}_",)

    for base in (
        sessions_dir(sid, repo_root=root),
        root / "experiment_game" / "data" / "sessions",
    ):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            if not any(d.name.startswith(p) for p in prefixes):
                continue
            found[d.resolve()] = d

    return sorted(found.values(), key=lambda p: p.name)


def phase_mode_to_board(phase_mode: Optional[str]) -> str:
    """phase_mode → 操作台板块标签（v1/v2/v3/v4/sim）。"""
    pm = str(phase_mode or "").strip()
    if pm == "v2_session":
        return "v2"
    if pm == "v3_session":
        return "v3"
    if pm == "v4_session":
        return "v4"
    if pm == "sim_v3_session":
        return "sim"
    if pm in ("phase2_full", "phase1", ""):
        return "v1"
    return "other"


def _session_seq_num(session_id: str) -> Optional[int]:
    """解析 w01 / ws01 / ses01 中的序号；无法解析则 None。"""
    m = re.match(r"^(?:w|ws|ses)(\d+)$", str(session_id or "").strip(), re.I)
    return int(m.group(1)) if m else None


def _read_session_phase_mode(session_dir: Path) -> Optional[str]:
    meta_path = Path(session_dir) / "session.meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pm = meta.get("phase_mode")
        return str(pm) if pm else None
    except Exception:
        return None


def suggest_session_id(
    subject_id: str,
    *,
    repo_root: Optional[Path] = None,
    phase_mode: Optional[str] = None,
    board: Optional[str] = None,
) -> str:
    """建议下一会话编号。

    默认按板块独立递增（``w01``、``w02``…）；``phase_mode`` / ``board`` 指定板块。
    兼容历史 ``ws##`` / ``ses##`` 计入该板块序号。未指定板块时回退为跨板块全局递增。
    """
    want_board = str(board or "").strip() or None
    if want_board is None and phase_mode is not None:
        want_board = phase_mode_to_board(phase_mode)

    nums: List[int] = []
    if want_board:
        for row in list_sessions(subject_id, repo_root=repo_root):
            if phase_mode_to_board(row.get("phase_mode")) != want_board:
                continue
            n = _session_seq_num(str(row.get("session_id") or ""))
            if n is not None:
                nums.append(n)
    else:
        for d in _discover_session_dirs(subject_id, repo_root=repo_root):
            if "_archived" in d.parts:
                continue
            _, sess, _ = _parse_session_name(d.name)
            n = _session_seq_num(sess)
            if n is not None:
                nums.append(n)
    nxt = (max(nums) + 1) if nums else 1
    return f"w{nxt:02d}"


def suggest_session_ids_by_board(
    subject_id: str,
    *,
    repo_root: Optional[Path] = None,
) -> Dict[str, str]:
    """各板块下一号：``{"v1":"w01","v2":"w01",...}``。"""
    return {
        b: suggest_session_id(subject_id, repo_root=repo_root, board=b)
        for b in ("v1", "v2", "v3", "v4")
    }


def list_dirs_for_session_id(
    subject_id: str,
    session_id: str,
    *,
    repo_root: Optional[Path] = None,
    phase_mode: Optional[str] = None,
    board: Optional[str] = None,
) -> List[Path]:
    """返回该被试下 session_id 匹配的已有会话目录（不含 _archived）。

    若提供 ``phase_mode`` / ``board``，仅返回同板块目录（同号跨板块可并存）。
    """
    sid = validate_subject_id(subject_id)
    want = str(session_id or "").strip().lower()
    if not want:
        return []
    want_board = str(board or "").strip() or None
    if want_board is None and phase_mode is not None:
        want_board = phase_mode_to_board(phase_mode)
    out: List[Path] = []
    for d in _discover_session_dirs(sid, repo_root=repo_root):
        if "_archived" in d.parts:
            continue
        _, sess, _ = _parse_session_name(d.name)
        if str(sess).lower() != want:
            continue
        if want_board is not None:
            pm = _read_session_phase_mode(d)
            if phase_mode_to_board(pm) != want_board:
                continue
        out.append(d)
    return out


def session_id_conflict(
    subject_id: str,
    session_id: str,
    *,
    repo_root: Optional[Path] = None,
    phase_mode: Optional[str] = None,
    board: Optional[str] = None,
) -> Dict[str, Any]:
    dirs = list_dirs_for_session_id(
        subject_id,
        session_id,
        repo_root=repo_root,
        phase_mode=phase_mode,
        board=board,
    )
    return {
        "subject_id": validate_subject_id(subject_id),
        "session_id": str(session_id or "").strip(),
        "phase_mode": phase_mode,
        "board": board or (phase_mode_to_board(phase_mode) if phase_mode else None),
        "exists": bool(dirs),
        "count": len(dirs),
        "dirs": [str(p) for p in dirs],
        "suggest_session_id": suggest_session_id(
            subject_id,
            repo_root=repo_root,
            phase_mode=phase_mode,
            board=board,
        ),
    }


def archive_sessions_for_id(
    subject_id: str,
    session_id: str,
    *,
    repo_root: Optional[Path] = None,
    phase_mode: Optional[str] = None,
    board: Optional[str] = None,
) -> List[str]:
    """覆盖前：将同 session_id（可选同板块）的旧目录移入 sessions/_archived/。"""
    root = Path(repo_root or _REPO)
    sid = validate_subject_id(subject_id)
    dirs = list_dirs_for_session_id(
        sid,
        session_id,
        repo_root=root,
        phase_mode=phase_mode,
        board=board,
    )
    if not dirs:
        return []
    sess_root = sessions_dir(sid, repo_root=root)
    arch = sess_root / "_archived"
    arch.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moved: List[str] = []
    for d in dirs:
        dest = arch / f"{d.name}__overwritten_{stamp}"
        # 同秒多次覆盖时加序号
        n = 1
        while dest.exists():
            dest = arch / f"{d.name}__overwritten_{stamp}_{n}"
            n += 1
        d.rename(dest)
        moved.append(str(dest))
    build_index(sid, repo_root=root)
    return moved


def _read_v3_report(session_dir: Path) -> Dict[str, Any]:
    p = session_dir / "v3_report.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_phase4_n_windows(session_dir: Path) -> Optional[int]:
    """汇总 session 内 phase4 切窗 manifest 的 n_windows（有则优先于试次估算）。"""
    total = 0
    found = False
    for sub in ("phase4_v2", "phase4_v2_game", "phase4"):
        manifest = session_dir / sub / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            nw = data.get("n_windows")
            if nw is not None:
                total += int(nw)
                found = True
        except Exception:
            continue
    return total if found else None


def estimate_ft_windows(
    n_trials: Optional[int],
    *,
    phase_mode: Optional[str] = None,
) -> Optional[int]:
    """粗估 FT 可用窗数（无 manifest 时按试次数 × 11）。"""
    if n_trials is None or int(n_trials) <= 0:
        return None
    n = int(n_trials)
    # v2 标定仅部分试次进 FT；用 0.65 折减避免高估
    if phase_mode == "v2_session":
        n = max(1, int(round(n * 0.65)))
    return n * OPENBMI_WINDOWS_PER_MI_TRIAL


def _session_metrics(session_dir: Path) -> Dict[str, Any]:
    session_dir = Path(session_dir)
    _, session_id, _ = _parse_session_name(session_dir.name)
    electrode = scan_session_electrodes(session_dir)
    report = _read_v3_report(session_dir)
    primary_acc = None
    window_acc = None
    valid_rate = None
    n_trials = None
    phase_mode = None
    record_excluded = False

    meta_path = session_dir / "session.meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            phase_mode = meta.get("phase_mode")
            n_trials = meta.get("trial_count")
            record_excluded = bool(meta.get("record_excluded"))
        except Exception:
            pass

    if report:
        overall = report.get("overall") or {}
        window_acc = overall.get("acc_window")
        if window_acc is None:
            window_acc = overall.get("acc_argmax_all_trials")
        primary_acc = overall.get("acc_argmax")
        if primary_acc is None and not overall:
            # 旧报告无 overall：回退到首块试次多数票
            blocks = report.get("blocks") or {}
            for blk in blocks.values():
                acc = (blk or {}).get("accuracy") or {}
                if acc.get("acc_argmax") is not None:
                    primary_acc = acc.get("acc_argmax")
                    break
        valid = report.get("valid_summary") or {}
        if valid.get("n_valid") and valid.get("n_total"):
            valid_rate = round(float(valid["n_valid"]) / float(valid["n_total"]), 3)

    has_eeg = _find_eeg_csv(session_dir) is not None
    ft_eligible = bool(electrode.get("electrode_ok") and has_eeg)

    n_windows = _read_phase4_n_windows(session_dir)
    n_windows_est = n_windows
    if n_windows_est is None:
        n_windows_est = estimate_ft_windows(n_trials, phase_mode=phase_mode)
    if n_windows_est is None:
        n_windows_est = EXP29_WINDOWS_PER_RUN

    return {
        "dir": session_dir.name,
        "path": str(session_dir.resolve()),
        "rel_path": rel_repo_path(session_dir),
        "session_id": session_id,
        "phase_mode": phase_mode,
        "n_trials": n_trials,
        "n_windows": n_windows,
        "n_windows_est": n_windows_est,
        "electrode_ok": electrode.get("electrode_ok", False),
        "electrode_warnings": electrode.get("warnings") or [],
        "primary_acc": primary_acc,
        "window_acc": window_acc,
        "valid_rate": valid_rate,
        "ft_eligible": ft_eligible and electrode.get("electrode_ok", False),
        "record_excluded": record_excluded,
    }


def list_sessions(subject_id: str, *, repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    dirs = _discover_session_dirs(subject_id, repo_root=repo_root)
    return [_session_metrics(d) for d in dirs]


def build_index(subject_id: str, *, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    sid = validate_subject_id(subject_id)
    sessions = list_sessions(sid, repo_root=repo_root)
    current = current_model_paths(sid, repo_root=repo_root)
    by_board = suggest_session_ids_by_board(sid, repo_root=repo_root)
    # 默认建议偏向 v3（探针常用）；前端会按当前板块再取 by_board
    default_sug = by_board.get("v3") or by_board.get("v2") or suggest_session_id(
        sid, repo_root=repo_root
    )
    idx = {
        "subject_id": sid,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "sessions": sessions,
        "current_model": current,
        "suggest_session_id": default_sug,
        "suggest_session_ids_by_board": by_board,
    }
    index_path = subject_root(sid, repo_root=repo_root) / "index.json"
    index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sub_json = subject_root(sid, repo_root=repo_root) / "subject.json"
    if sub_json.is_file():
        data = json.loads(sub_json.read_text(encoding="utf-8"))
        data["session_count"] = len(sessions)
        data["next_session_suggest"] = idx["suggest_session_id"]
        data["suggest_session_ids_by_board"] = by_board
        sub_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return idx


def current_model_paths(subject_id: str, *, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    sid = validate_subject_id(subject_id)
    root = Path(repo_root or _REPO)
    cur = models_current_dir(sid, repo_root=root)
    task = cur / "best_task.pt"
    three = cur / "best_three.pt"
    legacy = root / "experiment_game" / "data" / "models" / sid
    if not task.is_file() and (legacy / "best_task.pt").is_file():
        task = legacy / "best_task.pt"
        three = legacy / "best_three.pt"
        cur = legacy

    ok = task.is_file() and three.is_file()
    meta = {}
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
    }


def new_ft_run_dir(subject_id: str, *, repo_root: Optional[Path] = None) -> Path:
    sid = validate_subject_id(subject_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = models_ft_runs_dir(sid, repo_root=repo_root) / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def promote_ft_to_current(
    subject_id: str,
    ft_run_dir: Path,
    *,
    repo_root: Optional[Path] = None,
    reason: str = "",
) -> Dict[str, Any]:
    """将 ft_runs 快照复制到 models/current/。"""
    sid = validate_subject_id(subject_id)
    root = Path(repo_root or _REPO)
    src = Path(ft_run_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"FT 目录不存在: {src}")

    for name in ("best_task.pt", "best_three.pt"):
        if not (src / name).is_file():
            raise FileNotFoundError(f"缺少 {name} in {src}")

    cur = models_current_dir(sid, repo_root=root)
    from experiment_game.core.atomic_io import atomic_copy_files_into, atomic_write_json

    atomic_copy_files_into(
        src,
        cur,
        (
            "best_task.pt",
            "best_three.pt",
            "meta.json",
            "report.md",
            "release_gate.json",
            "e1f_overlay.json",
            "force_promote_warning.json",
        ),
    )
    from experiment_game.experiment.ft_promote_extras import promote_all4_extras

    all4_info = promote_all4_extras(src, cur, repo_root=root)

    promote_log = {
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "from_ft_run": rel_repo_path(src, repo_root=root),
        "reason": reason or "operator_confirmed",
        "all4": all4_info,
    }
    atomic_write_json(cur / "promote_log.json", promote_log)
    build_index(sid, repo_root=root)
    return {
        "ok": True,
        "current_dir": rel_repo_path(cur, repo_root=root),
        "weights": current_model_paths(sid, repo_root=root),
        "promote_log": promote_log,
        "all4": all4_info,
    }


def storage_paths_for_subject(subject_id: str, *, repo_root: Optional[Path] = None) -> Dict[str, str]:
    sid = validate_subject_id(subject_id)
    root = Path(repo_root or _REPO)
    sub_root = subject_root(sid, repo_root=root)
    return {
        "save_root": rel_repo_path(sessions_dir(sid, repo_root=root), repo_root=root),
        "subject_root": rel_repo_path(sub_root, repo_root=root),
    }
