"""加载 stieger_3s_hop100，按被试切成伪在线流。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import DATA_DIR, N_TIMES, OPENBMI_CHANS


@dataclass
class SubjectStream:
    subject_id: str
    X: np.ndarray  # (N,1,8,750) z-scored
    X_noz: np.ndarray | None
    y_task: np.ndarray
    y_three: np.ndarray
    trial_ids: np.ndarray  # Acc_paper 段键（Task/Rest 分 tid）
    cue_ids: np.ndarray  # 同一物理试次的配对 id（门控 / 半程划分）
    segs: np.ndarray  # mi | rest
    seg_keys: np.ndarray
    meta: dict[str, Any]


def _cue_ids_from_trial(trial_ids: np.ndarray, y_three: np.ndarray) -> np.ndarray:
    """Task=tid_base, Rest=tid_base+1 → cue_id = even floor。"""
    tid = np.asarray(trial_ids).astype(np.int64)
    y3 = np.asarray(y_three).astype(int)
    # Rest 窗用 tid-1 对齐 Task；无 Rest 的 Task 仍为自身
    cue = tid.copy()
    rest = y3 == 0
    cue[rest] = tid[rest] - 1
    return cue


def _require_data(data_dir: Path) -> None:
    need = [
        data_dir / "stieger_X.npy",
        data_dir / "stieger_y_task.npy",
        data_dir / "stieger_y_three.npy",
        data_dir / "stieger_subjects.npy",
        data_dir / "stieger_trial_id.npy",
    ]
    miss = [str(p) for p in need if not p.is_file()]
    if miss:
        raise FileNotFoundError(
            "缺少 stieger_3s_hop100 合并产物。请先跑:\n"
            "  cd code/preprocess_lab\n"
            "  python -m src.datasets.stieger.batch_3s_hop100\n"
            "缺失:\n  " + "\n  ".join(miss)
        )


_MERGED_CACHE: dict[str, dict[str, np.ndarray]] = {}


def load_merged(data_dir: Path | None = None) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir or DATA_DIR)
    cache_key = str(data_dir.resolve())
    cached = _MERGED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _require_data(data_dir)
    X = np.load(data_dir / "stieger_X.npy", mmap_mode="r")
    y_task = np.load(data_dir / "stieger_y_task.npy")
    y_three = np.load(data_dir / "stieger_y_three.npy")
    subjects = np.load(data_dir / "stieger_subjects.npy", allow_pickle=True)
    trial_id = np.load(data_dir / "stieger_trial_id.npy")
    noz_path = data_dir / "stieger_X_noz.npy"
    X_noz = np.load(noz_path, mmap_mode="r") if noz_path.is_file() else None
    if X.shape[-1] != N_TIMES:
        raise RuntimeError(f"期望 n_times={N_TIMES}，得到 {X.shape}")
    if X_noz is not None and X_noz.shape != X.shape:
        raise RuntimeError(f"X_noz shape {X_noz.shape} != X {X.shape}")
    pack = {
        "X": X,
        "X_noz": X_noz,
        "y_task": y_task,
        "y_three": y_three,
        "subjects": subjects,
        "trial_id": trial_id,
    }
    _MERGED_CACHE[cache_key] = pack
    return pack


def list_subjects(data_dir: Path | None = None) -> list[str]:
    pack = load_merged(data_dir)
    subs = sorted({str(s) for s in pack["subjects"].tolist()})
    return subs


def build_subject_stream(
    subject_id: str,
    *,
    data_dir: Path | None = None,
) -> SubjectStream:
    pack = load_merged(data_dir)
    sid = str(subject_id)
    mask = np.asarray([str(s) == sid for s in pack["subjects"].tolist()], dtype=bool)
    if not np.any(mask):
        raise KeyError(f"被试 {sid} 不在 {DATA_DIR}")
    X = pack["X"][mask]
    X_noz = pack["X_noz"][mask] if pack["X_noz"] is not None else None
    y_task = pack["y_task"][mask]
    y_three = pack["y_three"][mask]
    trial_ids = pack["trial_id"][mask]
    segs = np.where(y_three == 0, "rest", "mi").astype(object)
    cue_ids = _cue_ids_from_trial(trial_ids, y_three)
    seg_keys = np.asarray([f"{int(t)}:{s}" for t, s in zip(trial_ids, segs)], dtype=object)
    return SubjectStream(
        subject_id=sid,
        X=X,
        X_noz=X_noz,
        y_task=y_task,
        y_three=y_three,
        trial_ids=trial_ids,
        cue_ids=cue_ids,
        segs=segs,
        seg_keys=seg_keys,
        meta={
            "channels": list(OPENBMI_CHANS),
            "n_windows": int(len(X)),
            "n_mi": int((segs == "mi").sum()),
            "n_rest": int((segs == "rest").sum()),
            "has_x_noz": X_noz is not None and np.isfinite(X_noz).all(),
            "data_dir": str(Path(data_dir or DATA_DIR)),
        },
    )


def iter_subject_streams(
    *,
    data_dir: Path | None = None,
    subjects: list[str] | None = None,
):
    all_subs = list_subjects(data_dir)
    if subjects:
        allow = {s.strip() for s in subjects if s.strip()}
        all_subs = [s for s in all_subs if s in allow]
    for sid in all_subs:
        yield build_subject_stream(sid, data_dir=data_dir)
