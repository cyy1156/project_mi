"""会话异常/中断时的落盘兜底（重构阶段 2）。

保证尽量写出：
- session.meta.json（可含 aborted / incomplete）
- eeg.meta.json（或从 eeg.csv.meta.json 规范化）
- manifest.json（via finalize_session_layout）

可重复调用；调用方用标志位避免与正常收尾重复重活即可。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from experiment_game.core.atomic_io import atomic_write_json
from experiment_game.experiment.session import update_session_meta
from experiment_game.experiment.session_layout import finalize_session_layout


def ensure_eeg_meta(session_root: Path) -> Optional[Path]:
    """保证根目录有 eeg.meta.json；优先复制 eeg.csv.meta.json。"""
    root = Path(session_root)
    dest = root / "eeg.meta.json"
    src = root / "eeg.csv.meta.json"
    if dest.is_file():
        return dest
    if src.is_file():
        try:
            shutil.copy2(src, dest)
            return dest
        except OSError:
            pass
    # 无录制或尚未写出 sidecar：写最小 stub，便于下游发现「不完整」
    if (root / "eeg.csv").is_file():
        stub = {
            "source": "session_finalize_stub",
            "incomplete": True,
            "note": "eeg.csv 存在但缺少 recorder sidecar；由崩溃/中断兜底写入",
        }
        try:
            atomic_write_json(dest, stub)
            return dest
        except OSError:
            return None
    return None


def ensure_crash_artifacts(
    session_root: Path,
    *,
    aborted: bool = True,
    reason: str = "session_interrupted",
    acq_enabled: bool = True,
    save_layout: str = "phase_folders",
    save_continuous: bool = True,
    save_phase_slices: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """异常/强杀路径尽最大努力落盘。返回摘要供日志。"""
    root = Path(session_root)
    out: Dict[str, Any] = {"session_root": str(root), "ok": False}
    if not root.is_dir():
        out["error"] = "session_root_missing"
        return out

    meta_path = root / "session.meta.json"
    patch: Dict[str, Any] = {
        "aborted": bool(aborted),
        "abort_reason": str(reason),
        "incomplete": True,
    }
    if extra_meta:
        patch.update(extra_meta)
    try:
        if meta_path.is_file():
            update_session_meta(meta_path, **patch)
        else:
            atomic_write_json(
                meta_path,
                {
                    "session_dir": str(root),
                    **patch,
                    "note": "meta 缺失时由 session_finalize 兜底创建",
                },
            )
        out["session_meta"] = True
    except Exception as exc:  # noqa: BLE001
        out["session_meta_error"] = str(exc)

    try:
        eeg_meta = ensure_eeg_meta(root)
        out["eeg_meta"] = str(eeg_meta) if eeg_meta else None
    except Exception as exc:  # noqa: BLE001
        out["eeg_meta_error"] = str(exc)

    try:
        manifest = finalize_session_layout(
            root,
            save_layout=save_layout,
            save_continuous=save_continuous,
            save_phase_slices=save_phase_slices
            or str(save_layout) == "phase_folders",
            acq_enabled=bool(acq_enabled),
        )
        out["manifest"] = True
        out["events_parse_warnings"] = manifest.get("events_parse_warnings")
    except Exception as exc:  # noqa: BLE001
        out["manifest_error"] = str(exc)
        manifest = None

    # session_integrity 摘要（总册 §5.3 轻量落地；完整 EEG/events span 核对后续加强）
    try:
        integrity = {
            "event": "session_integrity",
            "manifest_ok": bool(out.get("manifest") and (root / "manifest.json").is_file()),
            "session_meta_ok": bool(out.get("session_meta")),
            "eeg_meta": out.get("eeg_meta"),
            "aborted": bool(aborted),
            "abort_reason": str(reason),
            "incomplete": True,
        }
        if isinstance(manifest, dict):
            integrity["events_parse_warnings"] = manifest.get("events_parse_warnings")
        atomic_write_json(root / "session_integrity.json", integrity)
        out["session_integrity"] = True
        try:
            update_session_meta(root / "session.meta.json", session_integrity=integrity)
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        out["session_integrity_error"] = str(exc)

    out["ok"] = bool(out.get("session_meta") and (root / "manifest.json").is_file())
    return out
