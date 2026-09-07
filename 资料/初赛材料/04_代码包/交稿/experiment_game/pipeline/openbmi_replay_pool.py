"""OpenBMI 3s/hop100 ReplayPool（fnz 离线 FT · 实验 27 统一策略 t0+0.10）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

from adapt_engine.ft import ReplayPool

_REPO = Path(__file__).resolve().parents[2]
OPENBMI_ROOT_ALL = _REPO / "code/preprocess_lab/out/openbmi_3s_hop100"
OPENBMI_ROOT_T0 = _REPO / "code/preprocess_lab/out/openbmi_3s_hop100_t0"

T0_SUBJS = {
    f"openbmi:subj{n:02d}"
    for n in (
        1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28,
        29, 30, 32, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47,
    )
}

DEFAULT_MAX_PER_CLASS = 3000
DEFAULT_REPLAY_RATIO = 0.10
DEFAULT_SEED = 42

_POOL_CACHE: Dict[tuple, ReplayPool] = {}
_TASK_POOL_CACHE: Dict[tuple, ReplayPool] = {}


def three_labels_to_task(labels: np.ndarray) -> np.ndarray:
    """三分类 OpenBMI 标签 → task 二分类：Rest=0，Left/Right=1。"""
    y = np.asarray(labels, dtype=np.int64)
    return np.where(y == 0, 0, 1).astype(np.int64)


def resolve_openbmi_root(*, prefer_t0: bool = True) -> Path:
    if prefer_t0 and (OPENBMI_ROOT_T0 / "openbmi_X.npy").is_file():
        return OPENBMI_ROOT_T0
    return OPENBMI_ROOT_ALL


def _collect_indices(
    root: Path,
    *,
    subject_allow: Optional[Set[str]] = None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> Dict[int, np.ndarray]:
    y = np.load(root / "openbmi_y_three.npy")
    subj = np.load(root / "openbmi_subjects.npy", allow_pickle=True)
    idx_by_class: Dict[int, list] = {0: [], 1: [], 2: []}
    for i in range(len(y)):
        if subject_allow is not None and str(subj[i]) not in subject_allow:
            continue
        idx_by_class[int(y[i])].append(i)

    rng = np.random.default_rng(seed)
    out: Dict[int, np.ndarray] = {}
    for c in (0, 1, 2):
        arr = np.asarray(idx_by_class[c], dtype=np.int64)
        if len(arr) > max_per_class:
            rng.shuffle(arr)
            arr = arr[:max_per_class]
        out[c] = arr
    return out


def _collect_task_indices(
    root: Path,
    *,
    subject_allow: Optional[Set[str]] = None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> Dict[int, np.ndarray]:
    """task 回放索引：0=Rest，1=Left+Right 合并（与 fnz y_task 一致）。"""
    y = np.load(root / "openbmi_y_three.npy")
    subj = np.load(root / "openbmi_subjects.npy", allow_pickle=True)
    idx_by_class: Dict[int, list] = {0: [], 1: []}
    for i in range(len(y)):
        if subject_allow is not None and str(subj[i]) not in subject_allow:
            continue
        c = 0 if int(y[i]) == 0 else 1
        idx_by_class[c].append(i)

    rng = np.random.default_rng(seed)
    out: Dict[int, np.ndarray] = {}
    for c in (0, 1):
        arr = np.asarray(idx_by_class[c], dtype=np.int64)
        if len(arr) > max_per_class:
            rng.shuffle(arr)
            arr = arr[:max_per_class]
        out[c] = arr
    return out


def build_task_replay_pool(
    *,
    root: Optional[Path] = None,
    subject_allow: Optional[Set[str]] = None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> Optional[ReplayPool]:
    """与 build_replay_pool 同源窗；标签映射为 task 二分类。"""
    root = Path(root or resolve_openbmi_root(prefer_t0=True))
    x_path = root / "openbmi_X.npy"
    if not x_path.is_file():
        return None

    key = ("task", str(root.resolve()), tuple(sorted(subject_allow or [])), max_per_class, seed)
    if key in _TASK_POOL_CACHE:
        return _TASK_POOL_CACHE[key]

    idx_map = _collect_task_indices(
        root,
        subject_allow=subject_allow,
        max_per_class=max_per_class,
        seed=seed,
    )
    if any(len(idx_map[c]) == 0 for c in (0, 1)):
        return None

    pick = np.concatenate([idx_map[c] for c in (0, 1)])
    xmm = np.load(x_path, mmap_mode="r")
    y3 = np.load(root / "openbmi_y_three.npy")
    wins = np.stack([xmm[int(i), 0].astype(np.float32) for i in pick], axis=0)
    labs = three_labels_to_task(y3[pick])
    pool = ReplayPool(wins, labs, seed=seed)
    _TASK_POOL_CACHE[key] = pool
    return pool


def build_replay_pool(
    *,
    root: Optional[Path] = None,
    subject_allow: Optional[Set[str]] = None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> Optional[ReplayPool]:
    """从 OpenBMI npy 建回放池；每类上限 max_per_class。"""
    root = Path(root or resolve_openbmi_root(prefer_t0=True))
    x_path = root / "openbmi_X.npy"
    if not x_path.is_file():
        return None

    key = ("three", str(root.resolve()), tuple(sorted(subject_allow or [])), max_per_class, seed)
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]

    idx_map = _collect_indices(
        root,
        subject_allow=subject_allow,
        max_per_class=max_per_class,
        seed=seed,
    )
    if any(len(idx_map[c]) == 0 for c in (0, 1, 2)):
        return None

    pick = np.concatenate([idx_map[c] for c in (0, 1, 2)])
    xmm = np.load(x_path, mmap_mode="r")
    y = np.load(root / "openbmi_y_three.npy")
    wins = np.stack([xmm[int(i), 0].astype(np.float32) for i in pick], axis=0)
    labs = y[pick].astype(np.int64)
    pool = ReplayPool(wins, labs, seed=seed)
    _POOL_CACHE[key] = pool
    return pool


def build_t0_task_replay_pool(
    *,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> Optional[ReplayPool]:
    """T0-36 task 回放池（Rest / MI 类均衡，与 three 池同源）。"""
    root = resolve_openbmi_root(prefer_t0=True)
    allow = None if root == OPENBMI_ROOT_T0 else T0_SUBJS
    return build_task_replay_pool(
        root=root,
        subject_allow=allow,
        max_per_class=max_per_class,
        seed=seed,
    )


def build_t0_replay_pool(
    *,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> Optional[ReplayPool]:
    """T0-36 三分类回放池（优先 openbmi_3s_hop100_t0）。"""
    root = resolve_openbmi_root(prefer_t0=True)
    allow = None if root == OPENBMI_ROOT_T0 else T0_SUBJS
    return build_replay_pool(
        root=root,
        subject_allow=allow,
        max_per_class=max_per_class,
        seed=seed,
    )
