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


def test_scan_fnz_ws01():
    repo = Path(__file__).resolve().parents[3]
    ses = repo / "experiment_game/data/sessions/fnz_ws01_20260826_164149"
    if not ses.is_dir():
        pytest.skip("no fnz ws01")
    rep = scan_session_electrodes(ses)
    assert rep["n_samples"] > 0
    assert "CZ" in rep["channels"] or "channels" in rep
