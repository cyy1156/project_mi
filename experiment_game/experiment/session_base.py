"""会话公共底座（总册 W6 · SessionRunner 基类）。

- ``session_runner.SessionRunner``（Phase2）已继承本基类
- ``session_runners.V2/V3/V4SessionRunner`` 薄封装对应函数入口
- ``attach_eeg_health`` / ``SessionServices`` 为三会话共用
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.markers import MarkerPublisher
from experiment_game.experiment.ws_bridge import WsBridge

OnConsole = Callable[[str], None]


@dataclass
class SessionServices:
    """会话三件套：事件 / marker / WS。"""

    events: EventLogger
    markers: MarkerPublisher
    bridge: WsBridge
    on_console: OnConsole = lambda s: print(s, flush=True)


def attach_eeg_health(
    buf: Any,
    services: SessionServices,
    *,
    tag: str = "session",
    enabled: bool = True,
) -> Any:
    """挂接 runtime EEGBus + HealthTicker；disabled 时返回 None。"""
    if not enabled or buf is None:
        return None
    from experiment_game.runtime.eeg_health import EegHealthTicker, ensure_session_bus

    bus = ensure_session_bus(buf)
    return EegHealthTicker(
        bus,
        bridge=services.bridge,
        events=services.events,
        markers=services.markers,
        log=lambda m: services.on_console(f"[{tag}] {m}"),
    )


class SessionRunnerBase:
    """范式会话基类：持有 SessionServices；子类实现 run()。

    现网 ``SessionRunner``（Phase2）已继承；v2/v3/v4 函数入口逐步迁移。
    """

    def __init__(self, services: SessionServices) -> None:
        self.services = services
        self.aborted = False
        self.abort_reason: Optional[str] = None

    @property
    def events(self) -> EventLogger:
        return self.services.events

    @property
    def markers(self) -> MarkerPublisher:
        return self.services.markers

    @property
    def bridge(self) -> WsBridge:
        return self.services.bridge

    def log(self, msg: str) -> None:
        self.services.on_console(msg)

    def run(self) -> None:
        raise NotImplementedError
