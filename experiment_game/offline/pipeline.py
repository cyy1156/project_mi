"""会话 → (X, y_task, y_three) 主流程。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiment_game.offline.epochs import (
    collect_window_specs,
    cut_window_with_baseline,
    labels_from_spec,
    resample_to_1000,
    to_model_tensor,
    trial_zscore,
)
from experiment_game.offline.filters import car_then_filter
from experiment_game.offline.load_session import SessionEEG, load_session


@dataclass
class EpochBundle:
    X: np.ndarray  # (N, 1, 8, 1000)
    y_task: np.ndarray  # (N,)
    y_three: np.ndarray  # (N,)
    trial_ids: np.ndarray  # (N,)
    kinds: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        yt = self.y_task
        y3 = self.y_three
        return {
            "n": int(self.X.shape[0]),
            "X_shape": list(self.X.shape),
            "y_task_counts": {
                "0": int(np.sum(yt == 0)),
                "1": int(np.sum(yt == 1)),
            },
            "y_three_counts": {
                "0": int(np.sum(y3 == 0)),
                "1": int(np.sum(y3 == 1)),
                "2": int(np.sum(y3 == 2)),
            },
            "skipped": self.meta.get("skipped", []),
        }


def sanity_check(bundle: EpochBundle) -> None:
    X, yt, y3 = bundle.X, bundle.y_task, bundle.y_three
    if X.ndim != 4 or X.shape[1:] != (1, 8, 1000):
        raise AssertionError(f"X shape 期望 (N,1,8,1000)，得到 {X.shape}")
    if yt.shape != (X.shape[0],) or y3.shape != (X.shape[0],):
        raise AssertionError("标签长度与 N 不一致")
    if not np.isfinite(X).all():
        raise AssertionError("X 含 NaN/Inf")
    if set(np.unique(yt)).issubset({0, 1}) is False:
        raise AssertionError(f"y_task 非法: {np.unique(yt)}")
    if set(np.unique(y3)).issubset({0, 1, 2}) is False:
        raise AssertionError(f"y_three 非法: {np.unique(y3)}")
    if not np.all((y3 == 0) == (yt == 0)):
        raise AssertionError("y_three==0 必须与 y_task==0 一致")
    if not np.all((y3 > 0) == (yt == 1)):
        raise AssertionError("y_three>0 必须与 y_task==1 一致")


def preprocess_session(
    session_dir: Path | str | SessionEEG,
    *,
    phases: Optional[Sequence[str]] = ("acquire",),
    apply_filter: bool = True,
) -> EpochBundle:
    """
    默认：phase==acquire 的 mi_start/rest_start → 4s 窗。
    输出与 preprocess_lab 一致：(N,1,8,1000) + y_task + y_three。
    """
    if isinstance(session_dir, SessionEEG):
        session = session_dir
    else:
        session = load_session(session_dir)

    x = session.x
    if apply_filter:
        x = car_then_filter(x, session.fs)

    specs = collect_window_specs(session, phases=phases)
    xs: List[np.ndarray] = []
    y_task: List[int] = []
    y_three: List[int] = []
    trial_ids: List[int] = []
    kinds: List[str] = []
    skipped: List[Dict[str, Any]] = []

    for spec in specs:
        win = cut_window_with_baseline(x, spec.sample, session.fs)
        if win is None:
            skipped.append(
                {
                    "trial_id": spec.trial_id,
                    "kind": spec.kind,
                    "reason": "window_oob",
                    "sample": spec.sample,
                }
            )
            continue
        win = resample_to_1000(win, session.fs)
        if win.shape != (1000, 8):
            skipped.append(
                {
                    "trial_id": spec.trial_id,
                    "kind": spec.kind,
                    "reason": f"bad_shape_{win.shape}",
                }
            )
            continue
        win = trial_zscore(win)
        lab_t, lab_3 = labels_from_spec(spec)
        xs.append(win)
        y_task.append(lab_t)
        y_three.append(lab_3)
        trial_ids.append(spec.trial_id)
        kinds.append(spec.kind)

    if not xs:
        X = np.zeros((0, 1, 8, 1000), np.float32)
        empty = np.zeros((0,), np.int64)
        bundle = EpochBundle(
            X=X,
            y_task=empty,
            y_three=empty.copy(),
            trial_ids=empty.copy(),
            kinds=[],
            meta={
                "session_dir": str(session.session_dir or ""),
                "skipped": skipped,
                "n_specs": len(specs),
                "_eeg_path": session.meta.get("_eeg_path"),
                "_events_path": session.meta.get("_events_path"),
                "exclude_rejected": True,
            },
        )
        return bundle

    bundle = EpochBundle(
        X=to_model_tensor(xs),
        y_task=np.asarray(y_task, dtype=np.int64),
        y_three=np.asarray(y_three, dtype=np.int64),
        trial_ids=np.asarray(trial_ids, dtype=np.int64),
        kinds=kinds,
        meta={
            "session_dir": str(session.session_dir or ""),
            "subject_id": session.meta.get("subject_id"),
            "session_id": session.meta.get("session_id"),
            "fs": session.fs,
            "phases": list(phases) if phases else None,
            "n_specs": len(specs),
            "skipped": skipped,
            "ch_names": session.ch_names,
            "_eeg_path": session.meta.get("_eeg_path"),
            "_events_path": session.meta.get("_events_path"),
            "exclude_rejected": True,
        },
    )
    sanity_check(bundle)
    return bundle


def save_bundle(
    bundle: EpochBundle,
    out_dir: Path | str,
    *,
    prefix: str = "",
    also_train_val: bool = True,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """
    写出 npy + meta.json。
    also_train_val=True 时额外写 train_/val_ 供 train_lab 使用。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = prefix

    np.save(out / f"{p}X.npy", bundle.X)
    np.save(out / f"{p}y_task.npy", bundle.y_task)
    np.save(out / f"{p}y_three.npy", bundle.y_three)
    np.save(out / f"{p}trial_ids.npy", bundle.trial_ids)

    meta = {
        **bundle.meta,
        "summary": bundle.summary(),
        "kinds": bundle.kinds,
    }
    (out / f"{p}meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if also_train_val and bundle.X.shape[0] >= 2:
        rng = np.random.RandomState(seed)
        n = bundle.X.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val_idx = np.sort(idx[:n_val])
        train_idx = np.sort(idx[n_val:])
        np.save(out / "train_X.npy", bundle.X[train_idx])
        np.save(out / "train_y_task.npy", bundle.y_task[train_idx])
        np.save(out / "train_y_three.npy", bundle.y_three[train_idx])
        np.save(out / "val_X.npy", bundle.X[val_idx])
        np.save(out / "val_y_task.npy", bundle.y_task[val_idx])
        np.save(out / "val_y_three.npy", bundle.y_three[val_idx])
        split_info = {
            "seed": seed,
            "val_ratio": val_ratio,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        }
        (out / "split.json").write_text(
            json.dumps(split_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return out
