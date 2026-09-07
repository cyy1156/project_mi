"""W2：EEGBus 看门狗 + RingBuffer 挂接 + judge buf_age。"""

from __future__ import annotations

import time

import numpy as np

from experiment_game.experiment.inference_v2 import JUDGE_BUF_STALE_S, InferenceService, RingBuffer
from experiment_game.runtime.eeg_bus import EEGBus, health_ws_payload
from experiment_game.runtime.eeg_health import EegHealthTicker, ensure_session_bus


def test_ring_buffer_notifies_bus():
    bus = EEGBus(stall_s=2.0, lost_s=5.0)
    buf = RingBuffer(capacity_s=2.0)
    buf.attach_bus(bus)
    buf.push(np.zeros(8))
    assert bus.health().n_samples == 1
    assert bus.health().state == "ok"


def test_bus_stall_state():
    bus = EEGBus(stall_s=0.05, lost_s=0.2)
    bus.note_push(10)
    assert bus.poll_health().state == "ok"
    time.sleep(0.08)
    assert bus.poll_health().state == "stall"
    assert bus.should_announce_stall() is True
    assert bus.should_announce_stall() is False


def test_health_ticker_emits_stall():
    bus = EEGBus(stall_s=0.05, lost_s=5.0)
    seen = []
    events = []

    class B:
        def broadcast(self, msg):
            seen.append(msg)

    class E:
        def emit(self, name, **kw):
            events.append(name)

    class M:
        def push(self, payload, **kw):
            events.append(payload)

    ticker = EegHealthTicker(bus, bridge=B(), events=E(), markers=M(), period_s=0.0)
    bus.note_push(1)
    time.sleep(0.06)
    ticker.tick()
    assert "eeg_stall" in events
    assert any(m.get("type") == "health" for m in seen)


def test_ensure_session_bus_idempotent():
    buf = RingBuffer(capacity_s=1.0)
    assert ensure_session_bus(buf) is ensure_session_bus(buf)


def test_annotate_buf_age_soft_stale():
    buf = RingBuffer(capacity_s=2.0)
    buf.push(np.zeros(8))
    svc = InferenceService.__new__(InferenceService)
    svc.buffer = buf
    out = InferenceService._annotate_buf_age(svc, {"pred": 1})
    assert "buf_age_s" in out
    assert out["stale"] is False
    assert out["buf_age_s"] <= JUDGE_BUF_STALE_S + 0.5


def test_health_ws_payload_fields():
    bus = EEGBus()
    bus.note_push(3)
    p = health_ws_payload(bus.poll_health(), buf_age_s=0.2)
    assert p["type"] == "health"
    assert p["buf_age_s"] == 0.2
    assert p["n_samples"] == 3
