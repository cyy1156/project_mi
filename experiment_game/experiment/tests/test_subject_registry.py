"""被试工作区与电极扫描。"""

from pathlib import Path

import pytest

from experiment_game.experiment.session_electrode import scan_session_electrodes
from experiment_game.experiment.subject_registry import (
    login_subject,
    suggest_session_id,
    suggest_session_ids_by_board,
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
    assert "suggest_session_ids_by_board" in info
    assert info["suggest_session_ids_by_board"]["v2"].startswith("w")


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


def test_suggest_session_id_per_board(tmp_path):
    from experiment_game.experiment.subject_registry import (
        login_subject,
        sessions_dir,
    )

    login_subject("abctest", repo_root=tmp_path)
    sess = sessions_dir("abctest", repo_root=tmp_path)

    d_v3 = sess / "abctest_ws01_20260828_120000"
    d_v3.mkdir(parents=True)
    (d_v3 / "session.meta.json").write_text(
        '{"phase_mode":"v3_session"}', encoding="utf-8"
    )
    d_v4 = sess / "abctest_ws02_20260828_130000"
    d_v4.mkdir(parents=True)
    (d_v4 / "session.meta.json").write_text(
        '{"phase_mode":"v4_session"}', encoding="utf-8"
    )
    d_v2 = sess / "abctest_w01_20260828_140000"
    d_v2.mkdir(parents=True)
    (d_v2 / "session.meta.json").write_text(
        '{"phase_mode":"v2_session"}', encoding="utf-8"
    )

    # 各板块独立递增；历史 ws## 计入该板块序号
    assert suggest_session_id("abctest", repo_root=tmp_path, board="v3") == "w02"
    assert suggest_session_id("abctest", repo_root=tmp_path, board="v4") == "w03"
    assert suggest_session_id("abctest", repo_root=tmp_path, board="v2") == "w02"
    by = suggest_session_ids_by_board("abctest", repo_root=tmp_path)
    assert by["v3"] == "w02"
    assert by["v2"] == "w02"
    assert by["v4"] == "w03"
    assert by["v1"] == "w01"


def test_session_id_conflict_and_archive_board_scoped(tmp_path):
    from experiment_game.experiment.subject_registry import (
        archive_sessions_for_id,
        login_subject,
        session_id_conflict,
        sessions_dir,
        suggest_session_id,
    )

    login_subject("abctest", repo_root=tmp_path)
    sess = sessions_dir("abctest", repo_root=tmp_path)
    d_v3 = sess / "abctest_w07_20260828_120000"
    d_v3.mkdir(parents=True)
    (d_v3 / "session.meta.json").write_text(
        '{"phase_mode":"v3_session"}', encoding="utf-8"
    )
    d_v2 = sess / "abctest_w07_20260828_121000"
    d_v2.mkdir(parents=True)
    (d_v2 / "session.meta.json").write_text(
        '{"phase_mode":"v2_session"}', encoding="utf-8"
    )

    conflict_v3 = session_id_conflict(
        "abctest", "w07", repo_root=tmp_path, phase_mode="v3_session"
    )
    assert conflict_v3["exists"] is True
    assert conflict_v3["count"] == 1
    assert conflict_v3["suggest_session_id"] == "w08"

    # 覆盖 v3 不应动 v2 同号
    moved = archive_sessions_for_id(
        "abctest", "w07", repo_root=tmp_path, phase_mode="v3_session"
    )
    assert len(moved) == 1
    assert not d_v3.exists()
    assert d_v2.exists()
    conflict2 = session_id_conflict(
        "abctest", "w07", repo_root=tmp_path, phase_mode="v3_session"
    )
    assert conflict2["exists"] is False
    assert suggest_session_id("abctest", repo_root=tmp_path, board="v2") == "w08"
    assert suggest_session_id("abctest", repo_root=tmp_path, board="v3") == "w01"


def test_scan_fnz_ws01():
    repo = Path(__file__).resolve().parents[3]
    ses = repo / "experiment_game/data/sessions/fnz_ws01_20260826_164149"
    if not ses.is_dir():
        pytest.skip("no fnz ws01")
    rep = scan_session_electrodes(ses)
    assert rep["n_samples"] > 0
    assert "CZ" in rep["channels"] or "channels" in rep
