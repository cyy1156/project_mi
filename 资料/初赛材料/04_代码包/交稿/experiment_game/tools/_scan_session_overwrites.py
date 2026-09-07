#!/usr/bin/env python3
"""Scan cohort subjects for archived overwrites and duplicate session_ids."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "experiment_game" / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "experiment_game" / "tools"))

from run_leave_next_e1f_task_ramp import _list_v3_sessions  # noqa: E402

SUBJECTS = ("syj0828", "xjh0828", "cyy0830", "fnz0830", "wzr0830", "xj0830")
ROOT = _REPO / "experiment_game" / "data" / "subjects"

def eeg_stats(p: Path) -> tuple[int | None, float | None]:
    eeg = p / "eeg.csv"
    if not eeg.is_file():
        return None, None
    rows = 0
    t0 = t1 = None
    with eeg.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            rows += 1
            try:
                t = float(row[0])
            except Exception:
                continue
            if t0 is None:
                t0 = t
            t1 = t
    span = (t1 - t0) if t0 is not None and t1 is not None else None
    return rows, span

def sess_key(name: str) -> str | None:
    m = re.search(r"(?:ws|w)(\d+)", name)
    return f"w{m.group(1)}" if m else None

def meta_of(p: Path) -> dict:
    mp = p / "session.meta.json"
    if not mp.is_file():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}

def main() -> None:
    print("=== _archived · __overwritten_* ===\n")
    for sub in SUBJECTS:
        arch = ROOT / sub / "sessions" / "_archived"
        items = sorted(
            p for p in arch.iterdir()
            if p.is_dir() and "__overwritten_" in p.name
        ) if arch.is_dir() else []
        if not items:
            print(f"{sub}: —")
            continue
        print(f"{sub}: {len(items)}")
        for p in items:
            m = meta_of(p)
            rows, span = eeg_stats(p)
            print(
                f"  {p.name}\n"
                f"    sid={m.get('session_id')} phase={m.get('phase_mode')} "
                f"created={m.get('created_at')} eeg_rows={rows} eeg_s={span}"
            )
        print()

    print("=== active duplicates (same wNN/wsNN, multiple dirs) ===\n")
    for sub in SUBJECTS:
        sess_root = ROOT / sub / "sessions"
        by_key: dict[str, list[tuple]] = {}
        for d in sorted(sess_root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            k = sess_key(d.name)
            if not k:
                continue
            m = meta_of(d)
            rows, span = eeg_stats(d)
            by_key.setdefault(k, []).append(
                (d.name, m.get("phase_mode"), m.get("created_at"), rows, span)
            )
        dups = {k: v for k, v in by_key.items() if len(v) > 1}
        chosen = _list_v3_sessions(sub)
        if dups:
            print(f"{sub}:")
            for k, lst in sorted(dups.items()):
                pick = chosen.get(k.replace("w", "ws") if sub.endswith("0828") else k)
                # normalize key for 0828 ws vs w
                if sub.endswith("0828"):
                    pick_key = k.replace("w", "ws")
                else:
                    pick_key = k
                pick = chosen.get(pick_key) or chosen.get(k.replace("w", "ws"))
                for name, phase, created, rows, span in lst:
                    mark = " ← Leave-Next picks" if pick and pick.name == name else ""
                    print(
                        f"  {k}: {name} phase={phase} created={created} "
                        f"rows={rows} span={span}{mark}"
                    )
            print(f"  ramp uses: {sorted(chosen)}")
            print()

    print("=== v4 + v3 same session_id (not archived) ===\n")
    for sub in SUBJECTS:
        sess_root = ROOT / sub / "sessions"
        by_key: dict[str, list] = {}
        for d in sorted(sess_root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            k = sess_key(d.name)
            if not k:
                continue
            m = meta_of(d)
            by_key.setdefault(k, []).append((d.name, m.get("phase_mode")))
        for k, lst in sorted(by_key.items()):
            phases = {p for _, p in lst}
            if "v4_session" in phases and "v3_session" in phases:
                print(f"{sub} {k}:")
                for name, phase in lst:
                    print(f"  {name} ({phase})")
                print()

if __name__ == "__main__":
    main()
