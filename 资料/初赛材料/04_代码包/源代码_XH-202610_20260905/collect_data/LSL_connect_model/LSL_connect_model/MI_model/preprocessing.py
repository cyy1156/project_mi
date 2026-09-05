"""CSV 读取、MI 窄带通、滑窗切样本。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    CHANNEL_COLUMNS,
    EPOCH_SAMPLES,
    MI_BANDPASS_HIGH_HZ,
    MI_BANDPASS_LOW_HZ,
    SFREQ,
    WARMUP_SAMPLES,
    WIN_SAMPLES,
    WIN_STEP_SAMPLES,
)

CLASS_PREFIX_RE = re.compile(r"^(left|right|stop|front|back)(\d*)$", re.IGNORECASE)
BAD_SESSION_SUFFIX = "_no"


def is_bad_session_file(path: Path) -> bool:
    """文件名含 `_no` 标记的 session 视为不可用（如 front_no.csv）。"""
    return path.stem.lower().endswith(BAD_SESSION_SUFFIX)


def parse_class_from_stem(stem: str) -> str | None:
    if stem.lower().endswith(BAD_SESSION_SUFFIX):
        return None
    m = CLASS_PREFIX_RE.match(stem)
    return m.group(1).lower() if m else None


def list_session_files(data_path: Path, class_prefixes: List[str]) -> List[Path]:
    prefixes = {p.lower() for p in class_prefixes}
    files: List[Path] = []
    for path in sorted(data_path.glob("*.csv")):
        if is_bad_session_file(path):
            continue
        cls = parse_class_from_stem(path.stem)
        if cls in prefixes:
            files.append(path)
    return files


def balance_sessions_per_class(
    files: List[Path],
    class_prefixes: List[str],
    max_per_class: int | None = None,
) -> tuple[List[Path], dict]:
    """
    每类只保留相同数量的 CSV（取各类 session 数的最小值，按文件名排序取前 N 个）。

    返回 (选用文件列表, 元信息)。
    """
    by_class: Dict[str, List[Path]] = {p: [] for p in class_prefixes}
    for path in sorted(files, key=lambda p: p.name):
        cls = parse_class_from_stem(path.stem)
        if cls and cls in by_class:
            by_class[cls].append(path)

    counts = {cls: len(by_class[cls]) for cls in class_prefixes}
    if not counts or min(counts.values()) == 0:
        missing = [c for c, n in counts.items() if n == 0]
        raise RuntimeError(f"以下类别无可用 CSV: {missing}")

    n_use = min(counts.values())
    if max_per_class is not None:
        n_use = min(n_use, max_per_class)

    selected: List[Path] = []
    excluded: Dict[str, List[str]] = {}
    used: Dict[str, List[str]] = {}
    for cls in sorted(class_prefixes):
        pick = by_class[cls][:n_use]
        drop = by_class[cls][n_use:]
        selected.extend(pick)
        used[cls] = [p.name for p in pick]
        excluded[cls] = [p.name for p in drop]

    info = {
        "enabled": True,
        "sessions_per_class": n_use,
        "available_counts": counts,
        "used_files": used,
        "excluded_files": excluded,
    }
    return sorted(selected, key=lambda p: p.name), info


# 旧录制列名 → 当前 CHANNEL_COLUMNS
_LEGACY_COLUMN_ALIASES = {
    "CPZ3": "CP3",
    "CPZ4": "CP4",
}


def load_session_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (n_channels, n_samples) float64 与 lsl_time。"""
    df = pd.read_csv(path)
    rename = {
        old: new
        for old, new in _LEGACY_COLUMN_ALIASES.items()
        if old in df.columns and new not in df.columns
    }
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in CHANNEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少通道列: {missing}")
    eeg = df[CHANNEL_COLUMNS].to_numpy(dtype=np.float64).T
    times = df["lsl_time"].to_numpy(dtype=np.float64)
    return eeg, times


def apply_mi_bandpass(eeg: np.ndarray) -> np.ndarray:
    """8~30 Hz 带通；输入/输出 (n_channels, n_samples)。采集端滤波不再重复。"""
    import mne

    return mne.filter.filter_data(
        eeg,
        sfreq=SFREQ,
        l_freq=MI_BANDPASS_LOW_HZ,
        h_freq=MI_BANDPASS_HIGH_HZ,
        method="fir",
        verbose=False,
    )


def window_group_id(session_stem: str, win_start: int) -> str:
    center = win_start + WIN_SAMPLES // 2
    epoch_id = center // EPOCH_SAMPLES
    return f"{session_stem}_ep{epoch_id}"


def extract_windows(
    eeg: np.ndarray,
    session_stem: str,
    label: int,
    group_encoder: Dict[str, int],
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    去 warmup → 滤波 → 滑窗。
    返回窗口列表、标签列表、group 整数 id 列表。
    """
    if eeg.shape[1] <= WARMUP_SAMPLES + WIN_SAMPLES:
        return [], [], []

    usable = eeg[:, WARMUP_SAMPLES:]
    filtered = apply_mi_bandpass(usable)

    windows: List[np.ndarray] = []
    labels: List[int] = []
    groups: List[int] = []

    n_samples = filtered.shape[1]
    start = 0
    while start + WIN_SAMPLES <= n_samples:
        win = filtered[:, start : start + WIN_SAMPLES].astype(np.float32)
        gid_str = window_group_id(session_stem, start)
        if gid_str not in group_encoder:
            group_encoder[gid_str] = len(group_encoder)

        windows.append(win)
        labels.append(label)
        groups.append(group_encoder[gid_str])
        start += WIN_STEP_SAMPLES

    return windows, labels, groups


def session_amplitude_stats(eeg: np.ndarray) -> Dict[str, float]:
    warmup = eeg[:, :WARMUP_SAMPLES] if eeg.shape[1] >= WARMUP_SAMPLES else eeg
    usable = eeg[:, WARMUP_SAMPLES:] if eeg.shape[1] > WARMUP_SAMPLES else eeg
    abs_all = np.abs(eeg)
    abs_warmup = np.abs(warmup)
    abs_usable = np.abs(usable)
    return {
        "mean_abs": float(abs_all.mean()),
        "max_abs": float(abs_all.max()),
        "warmup_max_abs": float(abs_warmup.max()),
        "usable_max_abs": float(abs_usable.max()),
        "usable_mean_abs": float(abs_usable.mean()),
    }
