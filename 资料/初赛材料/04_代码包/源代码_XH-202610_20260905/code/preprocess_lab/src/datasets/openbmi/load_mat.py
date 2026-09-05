"""OpenBMI / Lee2019-MI · 读取 `.mat` → ContinuousEEG。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.common.eeg_types import ContinuousEEG

# 文件名：sess01_subj07_EEG_MI.mat
_FNAME_RE = re.compile(
    r"sess(?P<sess>\d+)_subj(?P<subj>\d+)_EEG_MI\.mat$",
    re.IGNORECASE,
)

# OpenBMI y_dec / class：1=right, 2=left → 映射为 BCI2a 约定 1=Left, 2=Right
_OPENBMI_TO_BCI2A_THREE = {
    "left": 1,
    "right": 2,
}


def parse_sess_subj(path: Path) -> tuple[str, str]:
    m = _FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"无法解析 OpenBMI 文件名: {path.name}")
    sess = f"sess{int(m.group('sess')):02d}"
    subj = f"subj{int(m.group('subj')):02d}"
    return sess, subj


def subject_key(subj: str) -> str:
    """方案 A：两场同人 → openbmi:subjNN。"""
    return f"openbmi:{subj}"


def _class_to_three(y_class: np.ndarray) -> np.ndarray:
    out = []
    for c in y_class.tolist():
        key = str(c).strip().lower()
        if key not in _OPENBMI_TO_BCI2A_THREE:
            raise ValueError(f"未知 OpenBMI 类别: {c!r}")
        out.append(_OPENBMI_TO_BCI2A_THREE[key])
    return np.asarray(out, dtype=np.int64)


def _block_to_eeg(
    block,
    *,
    subject: str,
    session: str,
    block_name: str,
) -> ContinuousEEG:
    fs = float(block.fs)
    ch_names = [str(c).strip() for c in np.asarray(block.chan).tolist()]
    x = np.asarray(block.x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(ch_names):
        raise ValueError(f"{block_name}: x shape={x.shape} vs n_chan={len(ch_names)}")
    t = np.asarray(block.t, dtype=np.int64).reshape(-1)
    y3 = _class_to_three(np.asarray(block.y_class))
    if len(t) != len(y3):
        raise ValueError(f"{block_name}: len(t)={len(t)} != len(y)={len(y3)}")
    # events: [cue_sample, event_id]；event_id 已是 BCI2a three 码（1=左,2=右）
    events = np.stack([t, y3], axis=1).astype(np.int64)
    artifacts = np.zeros((len(t),), dtype=np.int64)
    return ContinuousEEG(
        subject=subject,
        session=f"{session}:{block_name}",
        x=x,
        fs=fs,
        ch_names=ch_names,
        events=events,
        labels=y3.copy(),
        artifacts=artifacts,
    )


def load_openbmi_mat(
    mat_path: Path | str,
    *,
    blocks: tuple[str, ...] = ("EEG_MI_train",),
) -> list[ContinuousEEG]:
    """
    读取单个 OpenBMI MI mat。

    方案 §1.2：仅 ``EEG_MI_train``；不读 ``EEG_MI_test``。
    返回 list[ContinuousEEG]（长度 1）；subject = openbmi:subjNN（方案 A）。
    """
    mat_path = Path(mat_path)
    sess, subj = parse_sess_subj(mat_path)
    subject = subject_key(subj)
    raw = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "EEG_MI_train" not in raw:
        raise KeyError(f"{mat_path.name}: 缺少 EEG_MI_train")
    return [
        _block_to_eeg(
            raw["EEG_MI_train"],
            subject=subject,
            session=sess,
            block_name="EEG_MI_train",
        )
    ]