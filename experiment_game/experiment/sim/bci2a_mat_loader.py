"""读取 BCI2a Training .mat（单 run）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.io

from experiment_game.experiment.sim.bci2a_channels import EEG22, select_8ch


@dataclass
class Bci2aRunData:
    subject_id: str
    run_id: str
    run_index: int
    mat_path: str
    fs: float
    x8: np.ndarray  # (n_times, 8) µV
    cue_samples: np.ndarray  # (n_mi,) L/R cue 样点
    labels: np.ndarray  # (n_mi,) 1=left 2=right
    mat_trial_indices: np.ndarray
    n_times: int
    # Rest 试次：Cue 前静息段（label=0），与 OpenBMI-Align FT 同源
    rest_start_samples: np.ndarray  # (n_rest_slot,)
    rest_end_samples: np.ndarray  # (n_rest_slot,) 通常 = 对应 MI cue


def _parse_run_index(run_id: str) -> int:
    rid = str(run_id or "").strip().lower()
    if not rid.startswith("run"):
        raise ValueError(f"run_id 须为 runN 格式: {run_id}")
    return int(rid[3:])


def count_run_capacity(rd: Bci2aRunData) -> Tuple[int, int, int, int]:
    """返回 (n_left, n_right, n_rest, n_total_max)。"""
    n_l = int(np.sum(rd.labels == 1))
    n_r = int(np.sum(rd.labels == 2))
    n_rest = int(len(rd.rest_start_samples))
    return n_l, n_r, n_rest, n_l + n_r + n_rest


def load_bci2a_run(
    mat_path: Path | str,
    run_id: str,
    *,
    rest_s: float = 4.0,
) -> Bci2aRunData:
    """加载 mat 内指定 run：L/R MI trial + Cue 前 Rest 槽位。"""
    path = Path(mat_path)
    if not path.is_file():
        raise FileNotFoundError(f"mat 不存在: {path}")
    run_index = _parse_run_index(run_id)
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    data = mat.get("data")
    if data is None or run_index >= len(data):
        raise ValueError(f"mat 内无 run 索引 {run_index}: {path}")

    run = data[run_index]
    trial = np.atleast_1d(run.trial) if run.trial is not None else np.array([])
    if trial.size == 0:
        raise ValueError(f"run{run_index} 无 trial（校准 run）")

    x = np.asarray(run.X, dtype=np.float64)
    y = np.atleast_1d(run.y).astype(int)
    artifacts = np.atleast_1d(getattr(run, "artifacts", np.zeros_like(y))).astype(int)
    fs = float(run.fs)

    x8 = select_8ch(x[:, :22], list(EEG22))
    cues: List[int] = []
    labels: List[int] = []
    tidx: List[int] = []
    rest_starts: List[int] = []
    rest_ends: List[int] = []
    rest_len = int(round(rest_s * fs))

    for i, (samp, lab) in enumerate(zip(trial.astype(int), y)):
        if artifacts.size > i and int(artifacts[i]) == 1:
            continue
        if int(lab) not in (1, 2):
            continue
        cue = int(samp)
        cues.append(cue)
        labels.append(int(lab))
        tidx.append(i)
        rs = max(0, cue - rest_len)
        rest_starts.append(rs)
        rest_ends.append(cue)

    if not cues:
        raise ValueError(f"run{run_index} 无有效 L/R trial")

    subject = path.stem[:3].upper()
    return Bci2aRunData(
        subject_id=subject,
        run_id=f"run{run_index}",
        run_index=run_index,
        mat_path=str(path.resolve()),
        fs=fs,
        x8=x8,
        cue_samples=np.asarray(cues, dtype=int),
        labels=np.asarray(labels, dtype=int),
        mat_trial_indices=np.asarray(tidx, dtype=int),
        n_times=int(x8.shape[0]),
        rest_start_samples=np.asarray(rest_starts, dtype=int),
        rest_end_samples=np.asarray(rest_ends, dtype=int),
    )


def list_labeled_runs(mat_path: Path | str) -> List[str]:
    """返回 mat 内所有带 trial 的 run 名（run0…）。"""
    path = Path(mat_path)
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    out: List[str] = []
    data = mat.get("data")
    if data is None:
        return []
    for i, run in enumerate(np.atleast_1d(data)):
        trial = np.atleast_1d(run.trial) if run.trial is not None else np.array([])
        if trial.size > 0:
            out.append(f"run{i}")
    return out
