"""Save-4：会话 → Phase4 切窗（仅 acquire + 未 reject）并写回指针。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from experiment_game.offline.pipeline import preprocess_session, save_bundle

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def default_epochs_out(session_root: Path) -> Path:
    return _PKG_ROOT / "data" / "epochs" / session_root.name


def session_eeg_path(session_root: Path) -> str:
    cont = session_root / "continuous" / "eeg.csv"
    if cont.is_file():
        return str(cont)
    root = session_root / "eeg.csv"
    return str(root) if root.is_file() else ""


def write_phase4_pointer(
    session_root: Path,
    *,
    epochs_dir: Path,
    summary: Dict[str, Any],
    ok: bool,
    message: str = "",
) -> Path:
    """写入 99_summary/phase4_pointer.json。"""
    summary_dir = Path(session_root) / "99_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pointer = {
        "ok": ok,
        "message": message,
        "epochs_dir": str(epochs_dir),
        "phases": ["acquire"],
        "exclude_rejected": True,
        "train_eligible_node": "06_acquire",
        "summary": summary,
    }
    path = summary_dir / "phase4_pointer.json"
    path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 人读一行
    (summary_dir / "README.txt").write_text(
        "Phase4 切窗指针见 phase4_pointer.json；"
        "主训练仅 acquire 且未 reject。\n"
        f"epochs_dir={epochs_dir}\n"
        f"ok={ok} n={summary.get('n')}\n",
        encoding="utf-8",
    )
    return path


def run_phase4_for_session(
    session_root: Path | str,
    *,
    out_dir: Optional[Path | str] = None,
    phases: Optional[Sequence[str]] = ("acquire",),
    apply_filter: bool = True,
    also_train_val: bool = True,
    val_ratio: float = 0.2,
    seed: int = 42,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    默认只切 phase=acquire，并排除 trial_reject（由 collect_window_specs 保证）。
    成功后写 99_summary/phase4_pointer.json。
    """
    root = Path(session_root)
    if not root.is_absolute():
        base = Path(repo_root) if repo_root else _REPO_ROOT
        root = (base / root).resolve()
    if not root.is_dir():
        return {"ok": False, "message": f"会话目录不存在: {root}", "summary": {}}

    out = Path(out_dir) if out_dir else default_epochs_out(root)
    if not out.is_absolute():
        base = Path(repo_root) if repo_root else _REPO_ROOT
        out = (base / out).resolve()

    try:
        bundle = preprocess_session(
            root,
            phases=list(phases) if phases is not None else None,
            apply_filter=apply_filter,
        )
        save_bundle(
            bundle,
            out,
            also_train_val=also_train_val,
            val_ratio=val_ratio,
            seed=seed,
        )
        summary = bundle.summary()
        summary["epochs_dir"] = str(out)
        summary["session_dir"] = bundle.meta.get("session_dir")
        summary["eeg_source"] = bundle.meta.get("_eeg_path") or session_eeg_path(root)
        summary["events_source"] = bundle.meta.get("_events_path")
        summary["phases"] = list(phases) if phases else None
        summary["exclude_rejected"] = True
        summary["train_eligible_node"] = "06_acquire"

        ok = int(summary.get("n") or 0) > 0
        msg = "PHASE4_OK" if ok else "未切出任何窗（检查 acquire 事件与 EEG 覆盖）"
        pointer = write_phase4_pointer(
            root,
            epochs_dir=out,
            summary=summary,
            ok=ok,
            message=msg,
        )
        return {
            "ok": ok,
            "message": msg,
            "epochs_dir": str(out),
            "pointer": str(pointer),
            "summary": summary,
            "skipped": summary.get("skipped") or [],
        }
    except Exception as exc:  # noqa: BLE001
        try:
            write_phase4_pointer(
                root,
                epochs_dir=out,
                summary={},
                ok=False,
                message=str(exc),
            )
        except Exception:  # noqa: BLE001
            pass
        return {
            "ok": False,
            "message": str(exc),
            "epochs_dir": str(out),
            "summary": {},
        }
