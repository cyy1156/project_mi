"""会话侧 EEG 健康钩子：broadcast health / eeg_stall（总册 W2）。"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from experiment_game.runtime.eeg_bus import (
    DEFAULT_STALL_S,
    EEGBus,
    health_ws_payload,
)


def ensure_session_bus(buf: Any, *, stall_s: float = DEFAULT_STALL_S) -> EEGBus:
    """保证 RingBuffer 挂有 EEGBus；已有则复用。"""
    bus = getattr(buf, "_bus", None)
    if isinstance(bus, EEGBus):
        return bus
    bus = EEGBus(stall_s=float(stall_s))
    if hasattr(buf, "attach_bus"):
        buf.attach_bus(bus)
    else:
        buf._bus = bus  # noqa: SLF001
    return bus


class EegHealthTicker:
    """节流广播 health；首次 stall 写 events/markers。"""

    def __init__(
        self,
        bus: EEGBus,
        *,
        bridge: Any = None,
        events: Any = None,
        markers: Any = None,
        period_s: float = 1.0,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.bus = bus
        self.bridge = bridge
        self.events = events
        self.markers = markers
        self.period_s = float(period_s)
        self.log = log or (lambda _m: None)
        self._last_broadcast = 0.0
        self._resume_pending = False

    def tick(self, buf: Any = None) -> None:
        now = time.monotonic()
        h = self.bus.poll_health(now)
        age = float(h.gap_s)
        if self.bus.should_announce_stall():
            self.log(f"EEG stall：已 {age:.1f}s 无新样本（阈值 {self.bus.stall_s:.0f}s）")
            if self.events is not None:
                try:
                    self.events.emit(
                        "eeg_stall",
                        phase="runtime",
                        last_sample_t=h.last_sample_t,
                        gap_s=age,
                    )
                except Exception:  # noqa: BLE001
                    pass
            if self.markers is not None:
                try:
                    self.markers.push(f"eeg_stall|gap={age:.1f}")
                except Exception:  # noqa: BLE001
                    pass
            self._resume_pending = True
            if self.bridge is not None:
                try:
                    self.bridge.broadcast(health_ws_payload(h, buf_age_s=age))
                except Exception:  # noqa: BLE001
                    pass
            return

        if self._resume_pending and h.state == "ok":
            self._resume_pending = False
            self.bus.mark_resume()
            if self.events is not None:
                try:
                    self.events.emit("eeg_resume", phase="runtime", gap_s=age)
                except Exception:  # noqa: BLE001
                    pass
            if self.markers is not None:
                try:
                    self.markers.push("eeg_resume")
                except Exception:  # noqa: BLE001
                    pass

        if now - self._last_broadcast < self.period_s:
            return
        self._last_broadcast = now
        buf_age = None
        if buf is not None and hasattr(buf, "last_push_age_s"):
            try:
                buf_age = buf.last_push_age_s()
            except Exception:  # noqa: BLE001
                buf_age = age
        if self.bridge is not None:
            try:
                self.bridge.broadcast(health_ws_payload(h, buf_age_s=buf_age))
            except Exception:  # noqa: BLE001
                pass
