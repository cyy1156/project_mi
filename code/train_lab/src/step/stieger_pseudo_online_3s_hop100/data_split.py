"""cue 级时间对半 + 前半内 Val。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data import SubjectStream
from util_metrics import jsonable


@dataclass(frozen=True)
class CueSplit:
    all_cues: list[int]
    train_cues: list[int]
    eval_cues: list[int]
    ft_train_cues: list[int]
    ft_val_cues: list[int]
    n_all: int
    n_train: int
    n_eval: int
    val_ratio: float
    seed: int


def split_cues_half(cue_ids: list[int]) -> tuple[list[int], list[int]]:
    ids = sorted(int(t) for t in cue_ids)
    n_train = len(ids) // 2
    return ids[:n_train], ids[n_train:]


def split_train_val_cues(
    train_cues: list[int],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    ids = sorted(int(t) for t in train_cues)
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
        train = [val[-1]]
        val = val[:-1]
    return train, val


def build_cue_split(
    stream: SubjectStream,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> CueSplit:
    all_cues = sorted({int(c) for c in stream.cue_ids.tolist()})
    train_cues, eval_cues = split_cues_half(all_cues)
    ft_train, ft_val = split_train_val_cues(
        train_cues, val_ratio=val_ratio, seed=seed
    )
    return CueSplit(
        all_cues=all_cues,
        train_cues=train_cues,
        eval_cues=eval_cues,
        ft_train_cues=ft_train,
        ft_val_cues=ft_val,
        n_all=len(all_cues),
        n_train=len(train_cues),
        n_eval=len(eval_cues),
        val_ratio=val_ratio,
        seed=seed,
    )


def window_mask_for_cues(
    cue_ids: np.ndarray, allowed: list[int] | set[int]
) -> np.ndarray:
    allow = {int(t) for t in allowed}
    cids = np.asarray(cue_ids).astype(int).reshape(-1)
    return np.asarray([int(c) in allow for c in cids.tolist()], dtype=bool)


def assert_no_leakage(split: CueSplit) -> None:
    tr = set(split.train_cues)
    ev = set(split.eval_cues)
    if tr & ev:
        raise RuntimeError(f"train/eval cue 交集非空: {sorted(tr & ev)}")
    ft_tr = set(split.ft_train_cues)
    ft_va = set(split.ft_val_cues)
    if ft_tr & ft_va:
        raise RuntimeError(f"ft_train/ft_val 交集非空: {sorted(ft_tr & ft_va)}")
    if (ft_tr | ft_va) - tr:
        raise RuntimeError("ft_* 超出 train half")


def write_split_artifacts(stream: SubjectStream, split: CueSplit, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_leakage(split)
    man: dict[str, Any] = {
        "subject_id": stream.subject_id,
        "split_rule": "cue_order_half + val_ratio_within_train",
        **asdict(split),
        "n_windows": int(len(stream.X)),
        "meta": stream.meta,
    }
    path = out_dir / "split_manifest.json"
    path.write_text(json.dumps(jsonable(man), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
