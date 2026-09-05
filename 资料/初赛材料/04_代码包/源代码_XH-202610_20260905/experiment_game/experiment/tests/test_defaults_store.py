"""defaults_store：相对化落盘 + example 回退。"""

from __future__ import annotations

import json
from pathlib import Path

from experiment_game.experiment.defaults_store import (
    _relativize_for_disk,
    load_operator_defaults,
    save_operator_defaults,
)


def test_relativize_save_root_under_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    abs_root = repo / "experiment_game" / "data" / "subjects" / "demo" / "sessions"
    abs_root.mkdir(parents=True)
    cfg = {"storage": {"save_root": str(abs_root)}}
    out = _relativize_for_disk(cfg, repo_root=repo)
    assert out["storage"]["save_root"] == "experiment_game/data/subjects/demo/sessions"


def test_save_operator_defaults_writes_relative(tmp_path: Path):
    repo = tmp_path / "MI"
    pkg = repo / "experiment_game"
    cfg_dir = pkg / "config"
    cfg_dir.mkdir(parents=True)
    sess = pkg / "data" / "subjects" / "x" / "sessions"
    sess.mkdir(parents=True)
    path = cfg_dir / "operator_defaults.json"
    raw = {
        "schema_version": 2,
        "subject": {"subject_id": "x", "session_id": "ws01"},
        "acquisition": {
            "enabled": True,
            "board_mode": "synthetic",
            "serial_port": "COM3",
            "sample_rate_hz": 250,
            "channel_labels": ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"],
            "markers_lsl": False,
        },
        "experiment": {"phase_mode": "v3_session", "acquire_trials": 4},
        "storage": {
            "save_root": str(sess.resolve()),
            "save_events": True,
            "save_session_meta": True,
        },
        "ui": {},
    }
    ok, msg, _ = save_operator_defaults(raw, path, repo_root=repo)
    assert ok, msg
    disk = json.loads(path.read_text(encoding="utf-8"))
    sr = disk["storage"]["save_root"].replace("\\", "/")
    assert not Path(sr).is_absolute()
    assert "experiment_game/data/subjects" in sr


def test_load_falls_back_to_example(tmp_path: Path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    sess = tmp_path / "sessions"
    sess.mkdir()
    example = cfg_dir / "operator_defaults.example.json"
    example.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "subject": {"subject_id": "ex", "session_id": "ws01"},
                "acquisition": {
                    "enabled": True,
                    "board_mode": "synthetic",
                    "serial_port": "COM3",
                    "sample_rate_hz": 250,
                    "channel_labels": ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"],
                    "markers_lsl": False,
                },
                "experiment": {"phase_mode": "v3_session", "acquire_trials": 2},
                "storage": {
                    "save_root": str(sess),
                    "save_events": True,
                    "save_session_meta": True,
                },
                "ui": {},
            }
        ),
        encoding="utf-8",
    )
    missing = cfg_dir / "operator_defaults.json"
    assert not missing.is_file()
    cfg, err = load_operator_defaults(missing, repo_root=tmp_path)
    assert cfg["subject"]["subject_id"] == "ex"
    assert err is None
