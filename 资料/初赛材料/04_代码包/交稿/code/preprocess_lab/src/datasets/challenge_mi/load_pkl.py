"""读取挑战杯官方 block pickle，切出 trial。"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

# 官方标签：201→左0 · 202→右1 · 204→Rest2（忽略 240/241）
LABEL_MAP = {201: 0, 202: 1, 204: 2}
TRIAL_LEN = 750  # 3 s @ 250 Hz
N_EEG = 59

# OpenBMI / 采集模型输入序（轨 B）；官方无 CPz，用 Pz 填 CPz 槽
OPENBMI_8_ORDER = ("Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz")


def resolve_data_root(explicit: Path | None = None) -> Path:
    """定位 ``DATA/挑战杯运动想象赛题数据文件``（兼容控制台乱码目录名）。"""
    if explicit is not None:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p.resolve()

    repo = Path(__file__).resolve().parents[5]  # …/MI
    data = repo / "DATA"
    if not data.is_dir():
        raise FileNotFoundError(f"找不到 DATA: {data}")

    preferred = data / "挑战杯运动想象赛题数据文件"
    if preferred.is_dir() and (preferred / "train").is_dir():
        return preferred.resolve()

    for c in data.iterdir():
        if c.is_dir() and (c / "train").is_dir() and (c / "sample_submission.csv").is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"在 {data} 下找不到含 train/ + sample_submission.csv 的官方集目录"
    )


def load_block(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "data" not in obj:
        raise ValueError(f"非法 block pickle: {path}")
    return obj


def _norm_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def channel_index_map(ch_names: list[str] | tuple[str, ...]) -> dict[str, int]:
    return {_norm_name(n): i for i, n in enumerate(ch_names)}


def select_eeg59(data: np.ndarray, ch_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """data (n_sig, T) → eeg (59, T)。"""
    if data.shape[0] < N_EEG:
        raise ValueError(f"通道不足: {data.shape}")
    eeg = np.asarray(data[:N_EEG], dtype=np.float64)
    names = [str(x) for x in list(ch_names)[:N_EEG]]
    if len(names) != N_EEG:
        raise ValueError(f"ch_names 前 {N_EEG} 无效: len={len(names)}")
    return eeg, names


def select_openbmi8(
    data: np.ndarray, ch_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """按 OpenBMI 模型序抽 8 导；CPz 槽用官方 Pz。"""
    idx_map = channel_index_map(ch_names)
    rows: list[np.ndarray] = []
    names_out: list[str] = []
    for slot in OPENBMI_8_ORDER:
        key = _norm_name(slot)
        if key == "CPZ":
            if "CPZ" in idx_map:
                i = idx_map["CPZ"]
                names_out.append(str(ch_names[i]))
            elif "PZ" in idx_map:
                i = idx_map["PZ"]
                names_out.append("CPz<-Pz")
            else:
                raise KeyError("官方集缺少 CPz/Pz，无法填 OpenBMI CPz 槽")
        else:
            if key not in idx_map:
                raise KeyError(f"官方集缺少通道 {slot}")
            i = idx_map[key]
            names_out.append(str(ch_names[i]))
        rows.append(np.asarray(data[i], dtype=np.float64))
    return np.stack(rows, axis=0), names_out


def extract_trials_from_block(
    obj: dict[str, Any],
    *,
    channel_mode: str = "59",
) -> dict[str, Any]:
    """
    从单个 block 切出 trials。

    Returns
    -------
    X : (n, C, 750) float64 原始（未滤波）
    y : (n,) int64 或 None（test）
    meta : dict
    """
    data = np.asarray(obj["data"])
    ch_names = list(obj["ch_names"])
    fs = float(obj.get("srate", 250.0))
    if abs(fs - 250.0) > 1e-3:
        raise ValueError(f"期望 fs=250，收到 {fs}")

    has_trigger = data.shape[0] >= 65
    if has_trigger:
        eeg_full = data[:-1]  # 64 信号行
        trigger = np.asarray(data[-1]).reshape(-1)
    else:
        eeg_full = data
        trigger = None

    if eeg_full.shape[1] < TRIAL_LEN:
        raise ValueError(f"时间过短: {eeg_full.shape}")

    n_possible = eeg_full.shape[1] // TRIAL_LEN
    starts = [i * TRIAL_LEN for i in range(n_possible)]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    starts_kept: list[int] = []

    for start in starts:
        seg = eeg_full[:, start : start + TRIAL_LEN]
        if channel_mode in ("59", "a59"):
            x, used_names = select_eeg59(seg, ch_names)
        elif channel_mode in ("8", "b8"):
            x, used_names = select_openbmi8(seg, ch_names)
        else:
            raise ValueError(f"未知 channel_mode={channel_mode}")

        y: int | None = None
        if trigger is not None:
            win = trigger[start : start + TRIAL_LEN]
            codes = [int(v) for v in np.unique(win) if int(v) in LABEL_MAP]
            if not codes:
                # 无有效 MI/Rest 标签（可能仅 240/241）→ 跳过
                continue
            # 取窗内首次出现的有效码
            y = None
            for v in win:
                iv = int(v)
                if iv in LABEL_MAP:
                    y = LABEL_MAP[iv]
                    break
            if y is None:
                continue
            ys.append(int(y))

        xs.append(x)
        starts_kept.append(int(start))

    if not xs:
        raise RuntimeError("block 未切出任何 trial")

    X = np.stack(xs, axis=0)  # (n, C, 750)
    out: dict[str, Any] = {
        "X": X,
        "y": np.asarray(ys, dtype=np.int64) if ys else None,
        "starts": np.asarray(starts_kept, dtype=np.int64),
        "fs": fs,
        "channel_names": used_names,
        "personID": obj.get("personID"),
        "blockID": obj.get("blockID"),
        "has_label": bool(ys),
    }
    return out


def list_train_blocks(root: Path) -> list[Path]:
    paths = sorted((Path(root) / "train").glob("S*/block_*.pkl"))
    if len(paths) != 30:
        # 仍允许继续，但警告由调用方处理
        pass
    return paths


def list_test_blocks(root: Path) -> list[Path]:
    return sorted((Path(root) / "test").glob("S*/block_*.pkl"))


def subject_from_path(path: Path) -> str:
    # .../train/S01/block_1.pkl → S01
    return path.parent.name
