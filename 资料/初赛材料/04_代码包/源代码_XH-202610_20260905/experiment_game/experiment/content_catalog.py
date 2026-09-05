"""物品 / 场景目录与轮换（Phase 3）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    label_zh: str


@dataclass(frozen=True)
class SceneSpec:
    id: str
    label_zh: str


OBJECTS: Sequence[ObjectSpec] = (
    ObjectSpec("cup", "杯子"),
    ObjectSpec("bottle", "瓶子"),
    ObjectSpec("apple", "苹果"),
)

SCENES: Sequence[SceneSpec] = (
    SceneSpec("home_desk", "家庭桌面"),
    SceneSpec("hospital_desk", "医院桌面"),
    SceneSpec("school_desk", "学校桌面"),
)

OBJECT_IDS: List[str] = [o.id for o in OBJECTS]
SCENE_IDS: List[str] = [s.id for s in SCENES]


def object_for_pair(pair_index: int, pool: Sequence[str] | None = None) -> str:
    """pair_index 从 0 起：每完成一对（2 trial）换下一物。"""
    ids = list(pool) if pool is not None else OBJECT_IDS
    return ids[pair_index % len(ids)]


def scene_for_trial(trial_index0: int, pool: Sequence[str] | None = None) -> str:
    """trial_index0 从 0 起：每 10 trial 换景。"""
    ids = list(pool) if pool is not None else SCENE_IDS
    block = trial_index0 // 10
    return ids[block % len(ids)]


def zh_object(object_id: str) -> str:
    for o in OBJECTS:
        if o.id == object_id:
            return o.label_zh
    return object_id


def zh_scene(scene_id: str) -> str:
    for s in SCENES:
        if s.id == scene_id:
            return s.label_zh
    return scene_id


def schedule_acquire(
    n_trials: int,
    *,
    seed: int | None = None,
) -> List[Tuple[int, int, str, str]]:
    """返回 [(trial_id, label, object, scene), ...]。"""
    import random

    from experiment_game.experiment.trial_sm import build_label_schedule

    rng = random.Random(seed)
    labels = build_label_schedule(n_trials, rng)
    rows: List[Tuple[int, int, str, str]] = []
    for i, lab in enumerate(labels):
        obj = object_for_pair(i // 2)
        sc = scene_for_trial(i)
        rows.append((i + 1, int(lab), obj, sc))
    return rows
