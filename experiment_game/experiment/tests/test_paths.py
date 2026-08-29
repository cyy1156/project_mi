"""core.paths 单测。"""

from __future__ import annotations

from pathlib import Path

from experiment_game.core.paths import look_like_absolute_windows, resolve, to_stored


def test_to_stored_and_resolve_roundtrip(tmp_path: Path):
    repo = tmp_path / "MI"
    target = repo / "experiment_game" / "data" / "subjects" / "a"
    target.mkdir(parents=True)
    stored = to_stored(target, root=repo)
    assert stored == "experiment_game/data/subjects/a"
    assert resolve(stored, root=repo) == target.resolve()


def test_look_like_absolute_windows():
    assert look_like_absolute_windows(r"D:\MI\foo")
    assert look_like_absolute_windows("C:/x")
    assert not look_like_absolute_windows("experiment_game/data/subjects")
