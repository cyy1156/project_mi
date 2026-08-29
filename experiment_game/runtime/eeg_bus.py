"""EEGBus 单总线多订阅 + 流健康看门狗（总册 §2.2 / §5.2）。

现网双消费链（LSL Recorder + RingBuffer）迁移期：
- RingBuffer.attach_bus(bus) 在 push 时通知总线；
- 会话侧 poll_health / maybe_emit_stall 发 WS ``health`` 与 ``eeg_stall``；
- 全量「单写多订」替换 Recorder 仍属后续战役接线。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from experiment_game.runtime.board_source import StreamHealth

# 总册阈值：>2s L2 暂停提示；>5s L3（现网 abort 仍可由会话用 3s 超时）
DEFAULT_STALL_S = 2.0
DEFAULT_LOST_S = 5.0
# judge 软陈旧：缓冲年龄超过该值标 stale=true（不计分侧可排除）
DEFAULT_JUDGE_STALE_S = 1.0


class EegSubscriber(Protocol):
    def on_chunk(self, t_lsl: np.ndarray, x: np.ndarray) -> None: ...


@dataclass
class EEGBus:
    """内存环总线：多订阅者 fan-out + 单调时钟看门狗。"""

    subscribers: List[EegSubscriber] = field(default_factory=list)
    stall_s: float = DEFAULT_STALL_S
    lost_s: float = DEFAULT_LOST_S
    last_health: StreamHealth = field(
        default_factory=lambda: StreamHealth(last_sample_t=None, n_samples=0)
    )
    _n: int = 0
    _last_push_mono: Optional[float] = None
    _watch_mono: float = field(default_factory=time.monotonic)
    _stall_announced: bool = False

    def subscribe(self, sub: EegSubscriber) -> None:
        if sub not in self.subscribers:
            self.subscribers.append(sub)

    def unsubscribe(self, sub: EegSubscriber) -> None:
        self.subscribers = [s for s in self.subscribers if s is not sub]

    def note_push(self, n_samples: int, *, t_lsl_last: Optional[float] = None) -> None:
        """RingBuffer / 板卡在写入后调用（可不经 publish 扇出）。"""
        n = max(0, int(n_samples))
        self._n += n
        self._last_push_mono = time.monotonic()
        if self._stall_announced and n > 0:
            self._stall_announced = False
        self.last_health = StreamHealth(
            last_sample_t=t_lsl_last,
            n_samples=self._n,
            gap_s=0.0,
            state="ok",
        )

    def publish(
        self,
        t_lsl: np.ndarray,
        x: np.ndarray,
        *,
        count: bool = True,
    ) -> None:
        t_arr = np.asarray(t_lsl).reshape(-1)
        x_arr = np.asarray(x)
        n = int(t_arr.shape[0]) if t_arr.size else int(x_arr.shape[0]) if x_arr.ndim else 0
        t_last = float(t_arr[-1]) if t_arr.size else None
        if count:
            self.note_push(n, t_lsl_last=t_last)
        else:
            # 仅扇出：样本计数已由 RingBuffer.attach_bus 记过
            self.last_health = StreamHealth(
                last_sample_t=t_last if t_last is not None else self.last_health.last_sample_t,
                n_samples=self._n,
                gap_s=0.0,
                state="ok",
            )
        for sub in list(self.subscribers):
            sub.on_chunk(t_arr, x_arr)

    def age_s(self, now_mono: Optional[float] = None) -> float:
        now = time.monotonic() if now_mono is None else float(now_mono)
        if self._last_push_mono is not None and self._n > 0:
            return max(0.0, now - self._last_push_mono)
        return max(0.0, now - self._watch_mono)

    def poll_health(self, now_mono: Optional[float] = None) -> StreamHealth:
        age = self.age_s(now_mono)
        if age >= float(self.lost_s):
            state = "lost"
        elif age >= float(self.stall_s):
            state = "stall"
        else:
            state = "ok"
        h = StreamHealth(
            last_sample_t=self.last_health.last_sample_t,
            n_samples=self._n,
            gap_s=float(age),
            state=state,
        )
        self.last_health = h
        return h

    def health(self) -> StreamHealth:
        return self.poll_health()

    def should_announce_stall(self) -> bool:
        """首次进入 stall 区时返回 True（调用方写 eeg_stall 后标记已宣告）。"""
        h = self.poll_health()
        if h.state in ("stall", "lost") and not self._stall_announced:
            self._stall_announced = True
            return True
        return False

    def mark_resume(self) -> bool:
        """流恢复时若曾 stall，返回 True 以便写 eeg_resume。"""
        if self._stall_announced and self.poll_health().state == "ok":
            self._stall_announced = False
            return True
        return False


def health_ws_payload(h: StreamHealth, *, buf_age_s: Optional[float] = None) -> Dict[str, object]:
    """WS ``health`` 消息（ws_protocol v2）。"""
    out: Dict[str, object] = {
        "type": "health",
        "eeg_state": h.state,
        "last_gap_s": float(h.gap_s),
        "n_samples": int(h.n_samples),
        "last_sample_t": h.last_sample_t,
    }
    if buf_age_s is not None:
        out["buf_age_s"] = float(buf_age_s)
    return out


def attach_bus_to_ring(buffer: Any, bus: EEGBus) -> EEGBus:
    """把 EEGBus 挂到 RingBuffer：每次 push 通知 note_push。"""
    buffer.attach_bus(bus)
    return bus
