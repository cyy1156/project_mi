"""trial 对半 + 前半内 Val 划分；窗级 mask。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stream import EvalStream, SegmentMeta, save_stream_artifacts


@dataclass(frozen=True)
class TrialSplit:
    all_trials: list[int]
    train_trials: list[int]
    eval_trials: list[int]
    ft_train_trials: list[int]
    ft_val_trials: list[int]
    n_all: int
    n_train: int
    n_eval: int
    val_ratio: float
    seed: int


def list_valid_trial_ids(segments: list[SegmentMeta]) -> list[int]:
    ids = sorted({int(m.trial_id) for m in segments})
    return ids


def split_trials_half(trial_ids: list[int]) -> tuple[list[int], list[int]]:
    ids = sorted(int(t) for t in trial_ids)
    n_train = len(ids) // 2
    return ids[:n_train], ids[n_train:]


def split_train_val_trials(
    train_trial_ids: list[int],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    ids = sorted(int(t) for t in train_trial_ids)
    if not ids:
        return [], []
    rng = np.random.RandomState(seed)
    perm = ids.copy()
    rng.shuffle(perm)
    n_val = max(1, int(round(len(perm) * val_ratio))) if len(perm) > 1 else 0
    if n_val >= len(perm):
        n_val = max(0, len(perm) - 1)
    val = sorted(perm[:n_val])
    train = sorted(perm[n_val:])
    if not train and val:
        # 极端：把最后一个 val 挪回 train
        train = [val[-1]]
        val = val[:-1]
    return train, val


def build_trial_split(
    stream: EvalStream,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> TrialSplit:
    all_trials = list_valid_trial_ids(stream.segments)
    train_trials, eval_trials = split_trials_half(all_trials)
    ft_train, ft_val = split_train_val_trials(
        train_trials, val_ratio=val_ratio, seed=seed
    )
    return TrialSplit(
        all_trials=all_trials,
        train_trials=train_trials,
        eval_trials=eval_trials,
        ft_train_trials=ft_train,
        ft_val_trials=ft_val,
        n_all=len(all_trials),
        n_train=len(train_trials),
        n_eval=len(eval_trials),
        val_ratio=val_ratio,
        seed=seed,
    )


def window_mask_for_trials(
    trial_ids: np.ndarray, allowed: list[int] | set[int]
) -> np.ndarray:
    allow = {int(t) for t in allowed}
    tids = np.asarray(trial_ids).astype(int).reshape(-1)
    return np.asarray([int(t) in allow for t in tids.tolist()], dtype=bool)


def assert_no_leakage(split: TrialSplit) -> None:
    tr = set(split.train_trials)
    ev = set(split.eval_trials)
    if tr & ev:
        raise RuntimeError(f"train/eval trial 交集非空: {sorted(tr & ev)}")
    ft_tr = set(split.ft_train_trials)
    ft_va = set(split.ft_val_trials)
    if ft_tr & ft_va:
        raise RuntimeError(f"ft_train/ft_val 交集非空: {sorted(ft_tr & ft_va)}")
    if (ft_tr | ft_va) - tr:
        raise RuntimeError("ft_* 超出 train half")
    if ft_tr | ft_va != tr and split.n_train > 0:
        # 允许空 val 时全集在 train
        missing = tr - (ft_tr | ft_va)
        if missing:
            raise RuntimeError(f"train half 未覆盖: {sorted(missing)}")


def count_windows(stream: EvalStream, trials: list[int]) -> dict[str, int]:
    mask = window_mask_for_trials(stream.trial_ids, trials)
    segs = stream.segs[mask] if mask.any() else np.array([], dtype=object)
    return {
        "n_windows": int(mask.sum()),
        "n_mi_windows": int(np.sum(segs == "mi")) if len(segs) else 0,
        "n_rest_windows": int(np.sum(segs == "rest")) if len(segs) else 0,
        "n_segments": len(
            {str(k) for k in stream.seg_keys[mask].tolist()}
        )
        if mask.any()
        else 0,
    }


def split_manifest_dict(
    stream: EvalStream, split: TrialSplit
) -> dict[str, Any]:
    assert_no_leakage(split)
    return {
        "subject_id": stream.subject_id,
        "session_id": stream.session_id,
        "session_dir": stream.session_dir,
        "split_rule": "trial_order_half + val_ratio_within_train",
        "val_ratio": split.val_ratio,
        "seed": split.seed,
        "n_all_trials": split.n_all,
        "n_train_trials": split.n_train,
        "n_eval_trials": split.n_eval,
        "all_trials": split.all_trials,
        "train_trials": split.train_trials,
        "eval_trials": split.eval_trials,
        "ft_train_trials": split.ft_train_trials,
        "ft_val_trials": split.ft_val_trials,
        "counts": {
            "all": count_windows(stream, split.all_trials),
            "train_half": count_windows(stream, split.train_trials),
            "eval_half": count_windows(stream, split.eval_trials),
            "ft_train": count_windows(stream, split.ft_train_trials),
            "ft_val": count_windows(stream, split.ft_val_trials),
        },
        "leakage_check": "pass",
        "stream_meta": stream.meta,
    }


def write_split_artifacts(
    stream: EvalStream,
    split: TrialSplit,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_stream_artifacts(stream, out_dir)
    man = split_manifest_dict(stream, split)
    path = out_dir / "split_manifest.json"
    path.write_text(
        json.dumps(man, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    # 方便人工扫一眼
    (out_dir / "split_summary.txt").write_text(
        "\n".join(
            [
                f"subject={stream.subject_id}",
                f"n_all={split.n_all} n_train={split.n_train} n_eval={split.n_eval}",
                f"ft_train={len(split.ft_train_trials)} ft_val={len(split.ft_val_trials)}",
                f"eval_windows={man['counts']['eval_half']['n_windows']}",
                f"ft_train_windows={man['counts']['ft_train']['n_windows']}",
                f"ft_val_windows={man['counts']['ft_val']['n_windows']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
