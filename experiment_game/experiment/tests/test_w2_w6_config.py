"""CsvRecorderSubscriber + protocol config + WS dispatch smoke."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_game.core.config import deep_merge, load_layered_config
from experiment_game.core.paths import repo_root
from experiment_game.runtime.csv_recorder import CsvRecorderSubscriber
from experiment_game.runtime.eeg_bus import EEGBus


def test_csv_recorder_via_bus(tmp_path: Path):
    path = tmp_path / "eeg.csv"
    bus = EEGBus()
    rec = CsvRecorderSubscriber(path)
    rec.open()
    bus.subscribe(rec)
    t = np.asarray([1.0, 1.004], dtype=np.float64)
    x = np.zeros((2, 8), dtype=np.float64)
    x[:, 0] = [1.0, 2.0]
    bus.publish(t, x)
    rec.close()
    text = path.read_text(encoding="utf-8")
    assert "lsl_time" in text
    assert "FC3" in text
    assert rec.rows_written == 2
    lines = [ln for ln in text.strip().splitlines() if ln]
    assert len(lines) == 3  # header + 2


def test_publish_count_false_no_double():
    bus = EEGBus()
    bus.note_push(5)
    bus.publish(np.asarray([0.0]), np.zeros((1, 8)), count=False)
    assert bus.poll_health().n_samples == 5


def test_load_protocol_yaml():
    cfg = load_layered_config(root=repo_root())
    assert cfg.get("protocol_id") == "openbmi_align_v1"
    assert cfg["channels"]["order"][0] == "FC3"
    assert cfg["windowing"]["win_s"] == 3.0
    assert cfg["eeg_watchdog"]["stall_s"] == 2.0
    assert cfg["eeg_watchdog"]["abort_s"] == 5.0


def test_resolve_eeg_watchdog_defaults():
    from experiment_game.runtime.eeg_bus import resolve_eeg_watchdog

    wd = resolve_eeg_watchdog(root=repo_root())
    assert wd["stall_s"] == 2.0
    assert wd["abort_s"] == 5.0
    assert wd["abort_s"] >= wd["stall_s"]


def test_deep_merge():
    assert deep_merge({"a": 1, "b": {"c": 1}}, {"b": {"d": 2}}) == {
        "a": 1,
        "b": {"c": 1, "d": 2},
    }


def test_ws_dispatch_table_keys():
    from experiment_game.experiment.orchestrator import OperatorService

    svc = object.__new__(OperatorService)
    table = OperatorService._ws_dispatch_table(svc)
    for key in (
        "config_validate",
        "session_start",
        "operator_hello",
        "finetune_start",
        "sim_catalog",
    ):
        assert key in table


def test_run_config_eats_protocol():
    from experiment_game.experiment.run_config import merge_run_config

    cfg = merge_run_config(None)
    assert cfg["experiment"]["timing"]["mi_s"] == 4.0
    assert cfg["acquisition"]["channel_labels"][0] == "FC3"
    assert cfg["acquisition"]["sample_rate_hz"] == 250
    assert cfg.get("extensions", {}).get("protocol_id") == "openbmi_align_v1"

    overridden = merge_run_config({"experiment": {"timing": {"mi_s": 3.5}}})
    assert overridden["experiment"]["timing"]["mi_s"] == 3.5
    assert overridden["experiment"]["timing"]["rest_s"] == 4.0


def test_session_runner_inherits_base():
    from experiment_game.experiment.session_base import SessionRunnerBase
    from experiment_game.experiment.session_runner import SessionRunner

    assert issubclass(SessionRunner, SessionRunnerBase)


def test_v_session_runners_inherit_base():
    from experiment_game.experiment.session_base import SessionRunnerBase
    from experiment_game.experiment.session_runners import (
        V2SessionRunner,
        V3SessionRunner,
        V4SessionRunner,
    )

    assert issubclass(V2SessionRunner, SessionRunnerBase)
    assert issubclass(V3SessionRunner, SessionRunnerBase)
    assert issubclass(V4SessionRunner, SessionRunnerBase)
