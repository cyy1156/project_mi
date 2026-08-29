"""EEGBus 单总线多订阅（总册 §2.2 / W2）— 接口层。

完整接线（替换双消费链）战役期再合入；当前提供：
- 订阅者注册 API
- 与 ``inference_v2.RingBuffer`` 的适配说明
- 健康状态结构，供后续 ``health`` WS 广播复用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np

from experiment_game.runtime.board_source import StreamHealth


class EegSubscriber(Protocol):
    def on_chunk(self, t_lsl: np.ndarray, x: np.ndarray) -> None: ...


@dataclass
class EEGBus:
    """内存环总线骨架：多订阅者 fan-out；尚未替换现网 Recorder/推理路径。"""

    subscribers: List[EegSubscriber] = field(default_factory=list)
    last_health: StreamHealth = field(
        default_factory=lambda: StreamHealth(last_sample_t=None, n_samples=0)
    )
    _n: int = 0

    def subscribe(self, sub: EegSubscriber) -> None:
        if sub not in self.subscribers:
            self.subscribers.append(sub)

    def unsubscribe(self, sub: EegSubscriber) -> None:
        self.subscribers = [s for s in self.subscribers if s is not sub]

    def publish(self, t_lsl: np.ndarray, x: np.ndarray) -> None:
        self._n += int(t_lsl.shape[0]) if hasattr(t_lsl, "shape") else 0
        t_last = float(t_lsl[-1]) if len(t_lsl) else None
        self.last_health = StreamHealth(
            last_sample_t=t_last,
            n_samples=self._n,
            gap_s=0.0,
            state="ok",
        )
        for sub in list(self.subscribers):
            sub.on_chunk(t_lsl, x)

    def health(self) -> StreamHealth:
        return self.last_health


def health_ws_payload(h: StreamHealth) -> Dict[str, object]:
    """WS ``health`` 消息字段草案（ws_protocol v2）。"""
    return {
        "type": "health",
        "eeg_state": h.state,
        "last_gap_s": float(h.gap_s),
        "n_samples": int(h.n_samples),
        "last_sample_t": h.last_sample_t,
    }
