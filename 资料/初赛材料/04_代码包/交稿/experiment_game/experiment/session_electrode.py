"""会话 EEG 电极质量扫描（平线 / dead 通道）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 与 v4 / fnz 文档一致：Cz、CPz 平线高危
WATCH_CHANNELS = ("CZ", "CPZ")
FLAT_PTP_UV = 5.0  # 全程 peak-to-peak 低于此视为平线（µV）
FLAT_STD_UV = 1.0
SAT_UV = 4000.0  # 接近 int16 饱和


def _find_eeg_csv(session_dir: Path) -> Optional[Path]:
    for p in (session_dir / "eeg.csv", session_dir / "continuous" / "eeg.csv"):
        if p.is_file():
            return p
    return None


def scan_session_electrodes(session_dir: Path) -> Dict[str, Any]:
    """扫描单 session 通道质量。

    返回 electrode_ok、per_channel、warnings 列表。
    """
    session_dir = Path(session_dir)
    eeg_path = _find_eeg_csv(session_dir)
    if eeg_path is None:
        return {
            "electrode_ok": False,
            "reason": "no_eeg",
            "channels": {},
            "warnings": ["无 eeg.csv"],
        }

    labels: List[str] = []
    meta_path = session_dir / "session.meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            labels = list(meta.get("channel_labels") or [])
        except Exception:
            labels = []

    df = pd.read_csv(eeg_path, nrows=0)
    ch_cols = [c for c in df.columns if c.startswith("ch_")]
    named_cols = [c for c in df.columns if c not in ("lsl_time", "timestamp") and not c.startswith("ch_")]
    if not ch_cols and named_cols:
        ch_cols = named_cols
        labels = [c.upper() for c in ch_cols]
    if not labels and ch_cols:
        labels = [c.replace("ch_", "").upper() for c in ch_cols]

    usecols = ["lsl_time"] + ch_cols
    df = pd.read_csv(eeg_path, usecols=usecols)
    if len(df) < 50:
        return {
            "electrode_ok": False,
            "reason": "too_short",
            "channels": {},
            "warnings": ["EEG 样本过少"],
        }

    per_ch: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for i, col in enumerate(ch_cols):
        lab = labels[i] if i < len(labels) else col.replace("ch_", "").upper()
        x = df[col].to_numpy(dtype=np.float64)
        ptp = float(np.ptp(x))
        std = float(np.std(x))
        mn = float(np.mean(x))
        flat = ptp < FLAT_PTP_UV and std < FLAT_STD_UV
        saturated = ptp > SAT_UV or np.max(np.abs(x)) > SAT_UV
        entry = {
            "ptp_uv": round(ptp, 3),
            "std_uv": round(std, 3),
            "mean_uv": round(mn, 3),
            "flat": bool(flat),
            "saturated": bool(saturated),
        }
        per_ch[lab] = entry
        if flat and lab in WATCH_CHANNELS:
            warnings.append(f"{lab} 平线 (ptp={ptp:.2f}µV)")
        if saturated and lab in WATCH_CHANNELS:
            warnings.append(f"{lab} 饱和/大 DC (ptp={ptp:.0f}µV)")

    watch_bad = any(
        per_ch.get(ch, {}).get("flat") or per_ch.get(ch, {}).get("saturated")
        for ch in WATCH_CHANNELS
        if ch in per_ch
    )
    electrode_ok = not watch_bad and not warnings

    return {
        "electrode_ok": electrode_ok,
        "reason": "ok" if electrode_ok else "watch_channels",
        "channels": per_ch,
        "warnings": warnings,
        "eeg_path": str(eeg_path),
        "n_samples": int(len(df)),
    }
