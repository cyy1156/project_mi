"""BoardSource 数据源协议（总册 §4.1 / W2 接口预留）。

真机 Cyton / synthetic / BCI2a replay 应实现本协议；
现网仍走 ``acquisition.service.AcquisitionFacade``，迁移期双轨。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class StreamHealth:
    last_sample_t: Optional[float]
    n_samples: int
    gap_s: float = 0.0
    state: str = "ok"  # ok | stall | lost


@dataclass
class BoardMeta:
    board_name: str
    sample_rate_hz: float
    channel_labels: List[str]
    firmware: str = ""


@runtime_checkable
class BoardSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def health(self) -> StreamHealth: ...

    def meta(self) -> BoardMeta: ...
