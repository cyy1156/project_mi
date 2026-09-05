"""openbmi replay 池单测。"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from experiment_game.tools.openbmi_replay_pool import (
    DEFAULT_REPLAY_RATIO,
    build_t0_replay_pool,
    build_t0_task_replay_pool,
    resolve_openbmi_root,
    three_labels_to_task,
)


def test_t0_pool_builds():
    root = resolve_openbmi_root(prefer_t0=True)
    assert (root / "openbmi_X.npy").is_file()
    pool = build_t0_replay_pool(seed=42)
    assert pool is not None
    assert len(pool.windows) > 0
    assert set(map(int, pool.labels)).issubset({0, 1, 2})


def test_t0_task_pool_builds():
    pool = build_t0_task_replay_pool(seed=42)
    assert pool is not None
    assert set(map(int, pool.labels)) == {0, 1}
    assert len(pool.windows) > 0


def test_three_to_task_labels():
    import numpy as np

    y = np.array([0, 1, 2, 1, 0])
    assert list(three_labels_to_task(y)) == [0, 1, 1, 1, 0]


def test_default_replay_ratio():
    assert DEFAULT_REPLAY_RATIO == 0.10
