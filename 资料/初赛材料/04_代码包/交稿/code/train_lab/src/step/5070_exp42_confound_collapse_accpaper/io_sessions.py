"""????? EEG/???????? 3s/hop100 ???????????"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from paths import CHANS, FS, SUBJECTS

def session_key_from_dirname(name: str) -> Optional[str]:
    for part in name.split("_"):
        p = part.lower()
        if p.startswith("ws") and p[2:].isdigit():
            return p
        if p.startswith("w") and not p.startswith("ws") and p[1:].isdigit():
            return p
    return None

def list_v3_sessions(member_id: str) -> Dict[str, Path]:
    """member_id -> {wNN|wsNN: session_dir}???????????? eeg.csv?"""
    root = SUBJECTS / member_id / "sessions"
    idx_path = SUBJECTS / member_id / "index.json"
    exclude: set[str] = set()
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        for s in idx.get("sessions") or []:
            if s.get("record_excluded"):
                exclude.add(str(s.get("dir") or ""))
            pm = s.get("phase_mode")
            if pm and pm != "v3_session":
                exclude.add(str(s.get("dir") or ""))

    by: Dict[str, Path] = {}
    if not root.is_dir():
        return by
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if d.name in exclude:
            continue
        if member_id == "syj0828" and "124816" in d.name:
            continue
        if member_id in ("xjh0828", "fnz0828") and d.name.endswith("_152231"):
            continue
        if member_id == "ycx0831" and "_w06_" in d.name:
            continue
        meta = d / "session.meta.json"
        if meta.is_file():
            try:
                phase = str(json.loads(meta.read_text(encoding="utf-8")).get("phase_mode") or "")
            except Exception:
                phase = ""
            if phase and phase != "v3_session":
                continue
        eeg = d / "eeg.csv"
        if not eeg.is_file():
            eeg = d / "continuous" / "eeg.csv"
        if not eeg.is_file():
            continue
        ws = session_key_from_dirname(d.name)
        if not ws:
            continue
        prev = by.get(ws)
        if prev is None or d.name > prev.name:
            by[ws] = d
    return by

def find_eeg_csv(session_dir: Path) -> Path:
    for p in (session_dir / "eeg.csv", session_dir / "continuous" / "eeg.csv"):
        if p.is_file():
            return p
    raise FileNotFoundError(f"no eeg.csv: {session_dir}")

def load_eeg(session_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """?? (lsl_time[T], x[T,8])?"""
    df = pd.read_csv(find_eeg_csv(session_dir))
    t = df["lsl_time"].to_numpy(dtype=np.float64)
    cols = []
    for name in CHANS:
        if name in df.columns:
            cols.append(name)
        else:
            alt = next((c for c in df.columns if c.upper() == name.upper()), None)
            if alt is None:
                raise KeyError(f"missing channel {name} in {session_dir}")
            cols.append(alt)
    x = df[cols].to_numpy(dtype=np.float64)
    return t, x

def load_events(session_dir: Path) -> List[Dict[str, Any]]:
    p = session_dir / "events.jsonl"
    if not p.is_file():
        p = session_dir / "continuous" / "events.jsonl"
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows

def parse_trials(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """? events ?? Rest/MI ?????label: 0=Rest,1=Left,2=Right?"""
    trials: List[Dict[str, Any]] = []
    i = 0
    n = len(events)
    while i < n:
        ev = events[i]
        name = str(ev.get("event") or "")
        if name == "rest_start":
            t0 = float(ev["t_lsl"])
            t1 = None
            label = int(ev.get("label", 0) or 0)
            j = i + 1
            while j < n:
                e2 = events[j]
                n2 = str(e2.get("event") or "")
                if n2 == "rest_end":
                    t1 = float(e2["t_lsl"])
                    if "label" in e2:
                        label = int(e2.get("label") or 0)
                    break
                if n2 in ("rest_start", "mi_start", "session_end"):
                    break
                j += 1
            if t1 is not None and t1 - t0 >= 1.0:
                trials.append(
                    {
                        "kind": "rest",
                        "label": 0,
                        "t0": t0,
                        "t1": t1,
                        "trial_idx": len(trials),
                    }
                )
            i = j + 1
            continue
        if name == "mi_start":
            t0 = float(ev["t_lsl"])
            t1 = t0 + 4.0
            label = int(ev.get("label", -1) if ev.get("label") is not None else -1)
            for k in range(i - 1, max(-1, i - 20), -1):
                e2 = events[k]
                if str(e2.get("event")) == "cue" and e2.get("label") is not None:
                    label = int(e2["label"])
                    break
                if str(e2.get("event")) == "trial_start" and e2.get("label") is not None:
                    label = int(e2["label"])
                    break
            j = i + 1
            while j < n:
                e2 = events[j]
                if str(e2.get("event")) == "mi_end":
                    t1 = float(e2["t_lsl"])
                    break
                if str(e2.get("event")) in ("mi_start", "rest_start", "session_end"):
                    break
                j += 1
            if label in (1, 2) and t1 - t0 >= 2.0:
                trials.append(
                    {
                        "kind": "mi",
                        "label": label,
                        "t0": t0,
                        "t1": t1,
                        "trial_idx": len(trials),
                    }
                )
            i = j + 1
            continue
        i += 1
    return trials

def load_eeg_meta(session_dir: Path) -> Dict[str, Any]:
    for p in (session_dir / "eeg.meta.json", session_dir / "continuous" / "eeg.meta.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}
