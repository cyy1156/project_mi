"""加载 experiment_game 会话目录：eeg.csv + events.jsonl。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiment_game.offline.channels import reorder_to_target


@dataclass
class SessionEEG:
    """连续 EEG + 已解析事件。"""

    x: np.ndarray  # (n_times, 8) 已按 TARGET 排序，µV
    fs: float
    ch_names: List[str]
    lsl_time: np.ndarray  # (n_times,)
    events: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    session_dir: Optional[Path] = None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_eeg_csv(path: Path) -> tuple[np.ndarray, List[str], np.ndarray]:
    """返回 x(n,t ch), ch_names, lsl_time。"""
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if not header or header[0] != "lsl_time":
        raise ValueError(f"eeg.csv 首列应为 lsl_time: {path}")
    ch_names = header[1:]
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    lsl_time = data[:, 0].astype(np.float64)
    x = data[:, 1:].astype(np.float64)
    if x.shape[1] != len(ch_names):
        raise ValueError(
            f"列数不匹配: data={x.shape[1]} header_ch={len(ch_names)}"
        )
    return x, ch_names, lsl_time


def rejected_trial_ids(events: Sequence[Dict[str, Any]]) -> set[int]:
    out: set[int] = set()
    for e in events:
        if e.get("event") == "trial_reject":
            tid = e.get("trial_id")
            if tid is not None:
                out.add(int(tid))
    return out


def load_session(
    session_dir: Path | str,
    *,
    require_eeg: bool = True,
    prefer_continuous: bool = True,
) -> SessionEEG:
    """
    读取会话目录。
    需要：eeg.csv、events.jsonl（根目录或 continuous/）；session.meta.json 可选。
    phase_folders 布局优先 continuous/（与对齐金标准一致）。
    """
    root = Path(session_dir)
    cont = root / "continuous"
    if prefer_continuous and (cont / "eeg.csv").is_file():
        eeg_path = cont / "eeg.csv"
        events_path = (
            cont / "events.jsonl"
            if (cont / "events.jsonl").is_file()
            else root / "events.jsonl"
        )
    else:
        eeg_path = root / "eeg.csv"
        events_path = root / "events.jsonl"
        if not eeg_path.is_file() and (cont / "eeg.csv").is_file():
            eeg_path = cont / "eeg.csv"
        if not events_path.is_file() and (cont / "events.jsonl").is_file():
            events_path = cont / "events.jsonl"

    meta_path = root / "session.meta.json"

    if not events_path.is_file():
        raise FileNotFoundError(f"缺少 events.jsonl: {events_path}")
    events = _read_jsonl(events_path)

    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if not eeg_path.is_file():
        if require_eeg:
            raise FileNotFoundError(
                f"缺少 eeg.csv（该会话可能未开启采集）: {root / 'eeg.csv'}"
            )
        raise FileNotFoundError(eeg_path)

    x_raw, ch_raw, lsl_time = _load_eeg_csv(eeg_path)
    x, ch_names = reorder_to_target(x_raw, ch_raw)
    fs = float(meta.get("sample_rate_hz") or 250.0)

    return SessionEEG(
        x=x,
        fs=fs,
        ch_names=ch_names,
        lsl_time=lsl_time,
        events=events,
        meta={
            **meta,
            "_eeg_path": str(eeg_path),
            "_events_path": str(events_path),
        },
        session_dir=root,
    )


def time_to_sample(lsl_time: np.ndarray, t: float) -> int:
    """最近邻（searchsorted 左邻后择近）。"""
    if len(lsl_time) == 0:
        raise ValueError("空 lsl_time")
    i = int(np.searchsorted(lsl_time, t, side="left"))
    if i <= 0:
        return 0
    if i >= len(lsl_time):
        return len(lsl_time) - 1
    if abs(lsl_time[i] - t) < abs(t - lsl_time[i - 1]):
        return i
    return i - 1
