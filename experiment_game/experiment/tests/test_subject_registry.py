"""被试工作区与电极扫描。"""

from pathlib import Path

import pytest

from experiment_game.experiment.session_electrode import scan_session_electrodes
from experiment_game.experiment.subject_registry import (
    login_subject,
    suggest_session_id,
    validate_subject_id,
)


def test_validate_subject_id():
    assert validate_subject_id("fnz") == "fnz"
    assert validate_subject_id("FNZ") == "fnz"
    with pytest.raises(ValueError):
        validate_subject_id("")


def test_login_creates_dirs(tmp_path):
    info = login_subject("tst01", display_name="test", repo_root=tmp_path)
    root = tmp_path / "experiment_game/data/subjects/tst01"
    assert root.is_dir()
    assert (root / "sessions").is_dir()
    assert (root / "models/current").is_dir()
    assert info["subject_id"] == "tst01"
    assert "sessions" in info
    assert isinstance(info["sessions"], list)


def test_estimate_ft_windows():
    from experiment_game.experiment.subject_registry import (
        estimate_ft_windows,
        OPENBMI_WINDOWS_PER_MI_TRIAL,
    )

    assert estimate_ft_windows(36) == 36 * OPENBMI_WINDOWS_PER_MI_TRIAL
    assert estimate_ft_windows(0) is None
    v2 = estimate_ft_windows(36, phase_mode="v2_session")
    assert v2 is not None
    assert v2 < 36 * OPENBMI_WINDOWS_PER_MI_TRIAL


def test_suggest_session_id_ws():
    sid = suggest_session_id("fnz", repo_root=Path(__file__).resolve().parents[3])
    # fnz has ws01-ws03 in data/sessions
    assert sid.startswith("ws")


def test_session_id_conflict_and_archive(tmp_path):
    from experiment_game.experiment.subject_registry import (
        archive_sessions_for_id,
        login_subject,
        session_id_conflict,
        sessions_dir,
        suggest_session_id,
    )

    login_subject("abctest", repo_root=tmp_path)
    sess = sessions_dir("abctest", repo_root=tmp_path)
    d1 = sess / "abctest_ws07_20260828_120000"
    d1.mkdir(parents=True)
    (d1 / "session.meta.json").write_text('{"phase_mode":"v3_session"}', encoding="utf-8")

    conflict = session_id_conflict("abctest", "ws07", repo_root=tmp_path)
    assert conflict["exists"] is True
    assert conflict["count"] == 1
    assert conflict["suggest_session_id"] == "ws08"

    moved = archive_sessions_for_id("abctest", "ws07", repo_root=tmp_path)
    assert len(moved) == 1
    assert not d1.exists()
    assert Path(moved[0]).is_dir()
    conflict2 = session_id_conflict("abctest", "ws07", repo_root=tmp_path)
    assert conflict2["exists"] is False
    assert suggest_session_id("abctest", repo_root=tmp_path) == "ws01"


def test_scan_fnz_ws01():
    repo = Path(__file__).resolve().parents[3]
    ses = repo / "experiment_game/data/sessions/fnz_ws01_20260826_164149"
    if not ses.is_dir():
        pytest.skip("no fnz ws01")
    rep = scan_session_electrodes(ses)
    assert rep["n_samples"] > 0
    assert "CZ" in rep["channels"] or "channels" in rep
