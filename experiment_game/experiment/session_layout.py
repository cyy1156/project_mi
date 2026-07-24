"""会话落盘布局：continuous / by_phase / alignment（Save-1/2）。"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PHASE_NODES = [
    ("01_adapt", "adapt", None),
    ("02_learn_step1", "learn", 1),
    ("03_learn_step2", "learn", 2),
    ("04_learn_step3", "learn", 3),
    ("05_gate", "gate", None),
    ("06_acquire", "acquire", None),
]


def _read_events(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_events(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def slice_eeg_csv(
    src: Path,
    dst: Path,
    t0: float,
    t1: float,
) -> int:
    """按 lsl_time 半开区间 [t0, t1) 切片；返回写入行数。"""
    if not src.is_file():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8", newline="") as fin, dst.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "lsl_time" not in reader.fieldnames:
            return 0
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            t = float(row["lsl_time"])
            if t0 <= t < t1:
                writer.writerow(row)
                n += 1
    return n


def _node_window(
    events: List[Dict[str, Any]],
    phase: str,
    learn_step: Optional[int],
) -> Optional[Tuple[float, float]]:
    if phase == "gate":
        # learn phase_end → acquire phase_start
        t0 = None
        t1 = None
        for ev in events:
            if ev.get("event") == "phase_end" and ev.get("phase") == "learn":
                t0 = float(ev["t_lsl"])
            if ev.get("event") == "phase_start" and ev.get("phase") == "acquire":
                t1 = float(ev["t_lsl"])
                break
        if t0 is None and t1 is None:
            return None
        # 若跳过 learn，用 adapt end
        if t0 is None:
            for ev in events:
                if ev.get("event") == "phase_end" and ev.get("phase") == "adapt":
                    t0 = float(ev["t_lsl"])
        if t1 is None:
            for ev in events:
                if ev.get("event") == "session_end":
                    t1 = float(ev["t_lsl"])
        if t0 is not None and t1 is not None and t1 > t0:
            return t0, t1
        return None

    if learn_step is not None:
        t0 = t1 = None
        for ev in events:
            if (
                ev.get("event") == "learn_step_start"
                and int(ev.get("learn_step") or 0) == learn_step
            ):
                t0 = float(ev["t_lsl"])
            if (
                ev.get("event") == "learn_step_end"
                and int(ev.get("learn_step") or 0) == learn_step
            ):
                t1 = float(ev["t_lsl"])
        if t0 is not None and t1 is not None and t1 > t0:
            return t0, t1
        return None

    t0 = t1 = None
    for ev in events:
        if ev.get("event") == "phase_start" and ev.get("phase") == phase:
            t0 = float(ev["t_lsl"])
        if ev.get("event") == "phase_end" and ev.get("phase") == phase:
            t1 = float(ev["t_lsl"])
    if t0 is not None and t1 is not None and t1 > t0:
        return t0, t1
    return None


def _filter_events_window(
    events: List[Dict[str, Any]], t0: float, t1: float
) -> List[Dict[str, Any]]:
    out = []
    for ev in events:
        t = float(ev.get("t_lsl", 0))
        if t0 <= t <= t1:
            out.append(ev)
    return out


def finalize_session_layout(
    session_root: Path,
    *,
    save_layout: str = "flat",
    save_continuous: bool = True,
    save_phase_slices: bool = False,
    acq_enabled: bool = True,
) -> Dict[str, Any]:
    """
    在会话结束后整理目录。
    - 始终尽量写 alignment（由调用方或本函数触发 build）
    - continuous/: 复制根级 eeg/events
    - by_phase/: 若 save_phase_slices
    """
    session_root = Path(session_root)
    eeg_src = session_root / "eeg.csv"
    events_src = session_root / "events.jsonl"
    events = _read_events(events_src)

    manifest: Dict[str, Any] = {
        "session_root": str(session_root),
        "save_layout": save_layout,
        "acq_enabled": acq_enabled,
        "files": {},
    }

    cont = session_root / "continuous"
    if save_continuous or save_layout == "phase_folders":
        cont.mkdir(parents=True, exist_ok=True)
        _copy_file(events_src, cont / "events.jsonl")
        if acq_enabled and eeg_src.is_file():
            _copy_file(eeg_src, cont / "eeg.csv")
            meta_side = session_root / "eeg.csv.meta.json"
            if meta_side.is_file():
                _copy_file(meta_side, cont / "eeg.csv.meta.json")
        manifest["files"]["continuous"] = "continuous/"

    if save_phase_slices or save_layout == "phase_folders":
        by_phase = session_root / "by_phase"
        by_phase.mkdir(parents=True, exist_ok=True)
        nodes_meta = []
        for folder, phase, step in PHASE_NODES:
            win = _node_window(events, phase, step)
            node_dir = by_phase / folder
            node_dir.mkdir(parents=True, exist_ok=True)
            train_eligible = phase == "acquire"
            phase_meta: Dict[str, Any] = {
                "node": folder,
                "phase": phase,
                "learn_step": step,
                "train_eligible": train_eligible,
                "acq_enabled": acq_enabled,
                "files": {},
            }
            if win is None:
                phase_meta["t_start_lsl"] = None
                phase_meta["t_end_lsl"] = None
                phase_meta["note"] = "本会话未覆盖该节点（可能 skip）"
                (node_dir / "phase.meta.json").write_text(
                    json.dumps(phase_meta, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                nodes_meta.append(phase_meta)
                continue
            t0, t1 = win
            phase_meta["t_start_lsl"] = t0
            phase_meta["t_end_lsl"] = t1
            ev_sub = _filter_events_window(events, t0, t1)
            _write_events(node_dir / "events.jsonl", ev_sub)
            phase_meta["files"]["events"] = "events.jsonl"
            if acq_enabled and eeg_src.is_file():
                n = slice_eeg_csv(eeg_src, node_dir / "eeg.csv", t0, t1)
                phase_meta["files"]["eeg"] = "eeg.csv"
                phase_meta["eeg_rows"] = n
            (node_dir / "phase.meta.json").write_text(
                json.dumps(phase_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (node_dir / "README.txt").write_text(
                f"node={folder} phase={phase} train_eligible={train_eligible}\n",
                encoding="utf-8",
            )
            nodes_meta.append(phase_meta)
        manifest["files"]["by_phase"] = "by_phase/"
        manifest["phase_nodes"] = nodes_meta

    (session_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
