"""runtime EEGBus 骨架 + BoardSource 协议。"""

from __future__ import annotations

import numpy as np

from experiment_game.runtime.board_source import BoardSource, StreamHealth
from experiment_game.runtime.eeg_bus import EEGBus, health_ws_payload


class _Sub:
    def __init__(self) -> None:
        self.n = 0

    def on_chunk(self, t_lsl, x) -> None:
        self.n += 1


def test_eeg_bus_fanout():
    bus = EEGBus()
    a, b = _Sub(), _Sub()
    bus.subscribe(a)
    bus.subscribe(b)
    t = np.arange(10, dtype=np.float64)
    x = np.zeros((10, 8))
    bus.publish(t, x)
    assert a.n == 1 and b.n == 1
    assert bus.health().n_samples == 10
    payload = health_ws_payload(bus.health())
    assert payload["type"] == "health"


def test_board_source_protocol_shape():
    class Fake:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def health(self) -> StreamHealth:
            return StreamHealth(last_sample_t=1.0, n_samples=1)

        def meta(self):
            from experiment_game.runtime.board_source import BoardMeta

            return BoardMeta("synthetic", 250.0, ["FC3"])

    assert isinstance(Fake(), BoardSource)
