"""LiveEegCapture / RingBuffer LSL publish / CSV meta."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_game.experiment.inference_v2 import RingBuffer
from experiment_game.runtime.csv_recorder import CsvRecorderSubscriber
from experiment_game.runtime.eeg_bus import EEGBus
from experiment_game.runtime.eeg_health import ensure_session_bus


def test_ringbuffer_push_publishes_with_ts(tmp_path: Path):
    buf = RingBuffer()
    bus = ensure_session_bus(buf)
    rec = CsvRecorderSubscriber(tmp_path / "eeg.csv")
    rec.open()
    bus.subscribe(rec)
    t = np.asarray([10.0, 10.004], dtype=np.float64)
    x = np.zeros((2, 8), dtype=np.float64)
    x[:, 0] = [1.0, 2.0]
    buf.push(x, t_lsl=t)
    rec.close()
    assert rec.rows_written == 2
    assert bus.poll_health().n_samples == 2
    text = (tmp_path / "eeg.csv").read_text(encoding="utf-8")
    assert "10.000000" in text


def test_csv_write_meta(tmp_path: Path):
    path = tmp_path / "eeg.csv"
    rec = CsvRecorderSubscriber(path)
    rec.open()
    bus = EEGBus()
    bus.subscribe(rec)
    bus.publish(np.asarray([1.0, 2.0]), np.zeros((2, 8)))
    meta = rec.write_meta(sample_rate_hz=250.0, use_synthetic=True)
    rec.close()
    assert meta["samples_written"] == 2
    assert "quality" in meta
    assert (tmp_path / "eeg.meta.json").is_file()
    assert (tmp_path / "eeg.csv.meta.json").is_file()


def test_acq_start_record_csv_flag():
    from experiment_game.acquisition.service import AcquisitionFacade

    # 仅检查签名/默认属性，不启板卡
    acq = AcquisitionFacade(use_synthetic=True)
    assert acq._record_csv is True
