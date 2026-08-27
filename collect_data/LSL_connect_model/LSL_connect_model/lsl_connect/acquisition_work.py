"""
第 6 课：后台采集线程 — BrainFlow → 预处理 → LSL push。
供第 7 课 ServiceManager 调用 start / stop。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pylsl import StreamOutlet

from lsl_connect.board import BoardConfig, CytonBoard
from lsl_connect.cyton_link import probe_cyton_version
from lsl_connect.lsl_streams import (
    BoardToLslTimestampMapper,
    LslStreamConfig,
    create_outlets,
    push_accel_chunk,
    push_eeg_chunk,
)
from lsl_connect.preprocessing import (
    PreprocessConfig,
    StreamingEegFilter,
    counts_to_microvolts,
    preprocess_accel_batch,
)

LinkEventCallback = Callable[[Dict[str, Any]], None]


@dataclass
class AcquisitionConfig:
    """采集循环参数（对应 eeg_broadcaster 的 BUFFER_SIZE 等）。"""

    buffer_size: int = 25  # 单批 push 上限（LSL chunk 大小）
    loop_sleep_sec: float = 0.005
    stats_every_n_batches: int = 20
    quiet: bool = False
    # 真机无线断流检测与自动重连（合成板在 start() 里旁路）
    stall_detect_enabled: bool = True
    stall_threshold_sec: float = 4.0
    reconnect_max_attempts: int = 3
    reconnect_cooldown_sec: float = 5.0
    reconnect_cooldown_max_sec: float = 30.0


class AcquisitionWorker:
    """
    在独立线程中运行采集循环。

    用法:
        worker = AcquisitionWorker()
        worker.start()
        print(worker.get_samples_pushed())
        worker.stop()
    """

    def __init__(
        self,
        board_config: Optional[BoardConfig] = None,
        lsl_config: Optional[LslStreamConfig] = None,
        preprocess_config: Optional[PreprocessConfig] = None,
        acq_config: Optional[AcquisitionConfig] = None,
        on_link_event: Optional[LinkEventCallback] = None,
    ) -> None:
        self._board_config = board_config or BoardConfig(use_synthetic=True)
        self._lsl_config = lsl_config or LslStreamConfig(
            sample_rate=250,
            channel_count=self._board_config.cyton_eeg_count,
            use_synthetic=self._board_config.use_synthetic,
        )

        self._preprocess_config = preprocess_config or PreprocessConfig()
        self._acq_config = acq_config or AcquisitionConfig()
        self._on_link_event = on_link_event

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stats_lock = threading.Lock()
        self._samples_pushed = 0
        self._batch_count = 0

        self._board: Optional[CytonBoard] = None
        self._outlet_eeg: Optional[StreamOutlet] = None
        self._outlet_accel: Optional[StreamOutlet] = None
        self._eeg_channel: Optional[np.ndarray] = None
        self._accel_channel: Optional[np.ndarray] = None
        self._ts_channel: Optional[int] = None
        self._ts_mapper = BoardToLslTimestampMapper()
        self._stream_filter: Optional[StreamingEegFilter] = None

        self._stall_enabled = False
        self._last_data_at = 0.0
        self._last_reconnect_at: Optional[float] = None
        self._link_dead = False
        self._stall_count = 0
        self._reconnect_ok = 0
        self._reconnect_fail = 0
        self._link_events: List[Dict[str, Any]] = []

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_samples_pushed(self) -> int:
        with self._stats_lock:
            return self._samples_pushed

    def get_link_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                "stall_count": self._stall_count,
                "reconnect_ok": self._reconnect_ok,
                "reconnect_fail": self._reconnect_fail,
                "link_dead": self._link_dead,
                "last_event": self._link_events[-1] if self._link_events else None,
                "events": list(self._link_events[-10:]),
            }

    def _emit_link_event(self, kind: str, **fields: Any) -> None:
        ev: Dict[str, Any] = {"kind": kind, "at": time.time(), **fields}
        with self._stats_lock:
            self._link_events.append(ev)
            if len(self._link_events) > 50:
                self._link_events = self._link_events[-50:]
        if self._on_link_event is not None:
            try:
                self._on_link_event(ev)
            except Exception as exc:  # noqa: BLE001
                print(f"[警告] on_link_event: {exc}")

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("采集线程已在运行")

        self._stop_event.clear()
        with self._stats_lock:
            self._samples_pushed = 0
            self._batch_count = 0
            self._link_dead = False
            self._stall_count = 0
            self._reconnect_ok = 0
            self._reconnect_fail = 0
            self._link_events = []

        self._stall_enabled = (
            self._acq_config.stall_detect_enabled
            and not self._board_config.use_synthetic
        )
        self._last_data_at = time.monotonic()
        self._last_reconnect_at = None

        self._board = CytonBoard(self._board_config)
        self._board.connect()
        eeg, accel, ts = self._board.get_channel_indices()
        self._eeg_channel = eeg
        self._accel_channel = accel
        self._ts_channel = int(ts)

        self._ts_mapper.reset()

        # 因果流式滤波：状态跨批保持，取代旧的按批零相位（避免块边界伪迹）
        self._stream_filter = StreamingEegFilter.from_config(self._preprocess_config)

        n_eeg = len(self._eeg_channel)
        self._lsl_config.channel_count = n_eeg
        self._lsl_config.use_synthetic = self._board_config.use_synthetic
        self._lsl_config.sample_rate = self._preprocess_config.sample_rate
        self._outlet_eeg, self._outlet_accel = create_outlets(self._lsl_config)

        self._thread = threading.Thread(
            target=self._run_loop,
            name="AcquisitionWorker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 12.0) -> None:
        self._stop_event.set()

        if self._board is not None:
            self._board.stop_stream_only()
            if not self._board_config.use_synthetic:
                time.sleep(0.05)

        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                print("[警告] 采集线程未在超时内结束，重试 stop_stream 后再次等待")
                if self._board is not None:
                    self._board.stop_stream_only()
                    time.sleep(0.2)
                self._thread.join(timeout=3.0)
            self._thread = None

        if self._board is not None:
            self._board.disconnect()
            self._board = None

        self._ts_mapper.reset()
        self._stream_filter = None
        self._outlet_eeg = None
        self._outlet_accel = None

    def _push_slice(self, data: np.ndarray, fs: int) -> int:
        """处理并推送一批样本，返回推送样本数。"""
        assert self._outlet_eeg is not None
        assert self._outlet_accel is not None
        assert self._eeg_channel is not None
        assert self._accel_channel is not None
        assert self._ts_channel is not None

        board_ts = data[self._ts_channel, :]
        lsl_ts = self._ts_mapper.to_lsl_uniform(board_ts, fs)

        eeg_counts = data[self._eeg_channel, :]
        eeg_uv = counts_to_microvolts(eeg_counts)
        if self._preprocess_config.filter_enabled and self._stream_filter is not None:
            eeg_uv = self._stream_filter.process(eeg_uv)
        eeg_uv = eeg_uv.astype(np.float32)
        n = push_eeg_chunk(self._outlet_eeg, eeg_uv, timepstamps=lsl_ts)

        accel_ch = self._accel_channel
        if len(accel_ch) > 0 and int(accel_ch[0]) < data.shape[0]:
            accel_count = data[accel_ch, :]
            accel_ms2 = preprocess_accel_batch(accel_count)
            push_accel_chunk(self._outlet_accel, accel_ms2, timestamps=lsl_ts)

        return n

    def _reconnect_cooldown_sec(self) -> float:
        cfg = self._acq_config
        cooldown = float(cfg.reconnect_cooldown_sec)
        now = time.monotonic()
        if self._last_reconnect_at is not None and (now - self._last_reconnect_at) < 60.0:
            cooldown = min(float(cfg.reconnect_cooldown_max_sec), cooldown * 2.0)
        return cooldown

    def _handle_stall(self, now: float) -> None:
        assert self._board is not None
        cfg = self._acq_config
        port = self._board_config.serial_port
        gap_s = max(0.0, now - self._last_data_at)

        with self._stats_lock:
            self._stall_count += 1

        self._emit_link_event(
            "stall",
            gap_s=round(gap_s, 2),
            message=f"无线数据停滞 {gap_s:.1f}s",
        )
        if not cfg.quiet:
            print(f"[链路] stall {gap_s:.1f}s，开始自动重连…")

        for attempt in range(1, int(cfg.reconnect_max_attempts) + 1):
            if self._stop_event.is_set():
                return

            self._emit_link_event(
                "reconnect_attempt",
                attempt=attempt,
                max_attempts=cfg.reconnect_max_attempts,
                message=f"重连 {attempt}/{cfg.reconnect_max_attempts}",
            )
            if not cfg.quiet:
                print(f"[链路] 重连 {attempt}/{cfg.reconnect_max_attempts}…")

            try:
                self._board.stop_stream_only()
                self._board.disconnect()
            except Exception as exc:  # noqa: BLE001
                self._emit_link_event("reconnect_error", attempt=attempt, error=str(exc))

            probe = probe_cyton_version(port)
            if not probe.ok:
                self._emit_link_event(
                    "reconnect_probe_fail",
                    attempt=attempt,
                    failure_kind=probe.failure_kind.value,
                    message=probe.summary(),
                )
                if self._stop_event.wait(self._reconnect_cooldown_sec()):
                    return
                continue

            try:
                self._board.connect()
                self._ts_mapper.reset()
                if self._stream_filter is not None:
                    self._stream_filter.reset()
                self._last_data_at = time.monotonic()
                self._last_reconnect_at = self._last_data_at
                with self._stats_lock:
                    self._reconnect_ok += 1
                self._emit_link_event(
                    "reconnect_ok",
                    attempt=attempt,
                    message=f"重连成功（第 {attempt} 次）",
                )
                if not cfg.quiet:
                    print(f"[链路] 重连成功（第 {attempt} 次）")
                return
            except Exception as exc:  # noqa: BLE001
                self._emit_link_event(
                    "reconnect_fail",
                    attempt=attempt,
                    error=str(exc),
                    message=str(exc),
                )
                if self._stop_event.wait(self._reconnect_cooldown_sec()):
                    return

        with self._stats_lock:
            self._reconnect_fail += 1
            self._link_dead = True
        self._emit_link_event(
            "link_dead",
            message="无线断流，自动重连失败：请检查 Cyton 电量、dongle 距离后重开机",
        )
        if not cfg.quiet:
            print("[链路] 自动重连已达上限，link_dead=True")

    def _run_loop(self) -> None:
        assert self._board is not None

        cfg = self._acq_config
        fs = self._preprocess_config.sample_rate
        bs = max(1, cfg.buffer_size)

        if not cfg.quiet:
            print("-" * 50)
            print("AcquisitionWorker 运行中... 调用 stop() 结束")
            print(
                f"拉数: fetch_new_batch | push 块大小: {bs} | "
                f"滤波: {'ON' if self._preprocess_config.filter_enabled else 'OFF'}"
            )
            if self._stall_enabled:
                print(
                    f"断流检测: ON（>{cfg.stall_threshold_sec}s 触发重连，"
                    f"最多 {cfg.reconnect_max_attempts} 次）"
                )
            print("-" * 50)

        while not self._stop_event.is_set():
            now = time.monotonic()
            try:
                data = self._board.fetch_new_batch()
            except Exception as exc:  # noqa: BLE001
                if (
                    self._stall_enabled
                    and not self._link_dead
                    and not self._stop_event.is_set()
                ):
                    self._emit_link_event("fetch_error", error=str(exc))
                    self._handle_stall(now)
                time.sleep(cfg.loop_sleep_sec)
                continue

            if data.shape[1] > 0:
                self._last_data_at = now
            elif (
                self._stall_enabled
                and not self._link_dead
                and not self._stop_event.is_set()
                and now - self._last_data_at > cfg.stall_threshold_sec
            ):
                self._handle_stall(now)
                time.sleep(cfg.loop_sleep_sec)
                continue

            if data.shape[1] == 0:
                time.sleep(cfg.loop_sleep_sec)
                continue

            n_total = data.shape[1]
            for start in range(0, n_total, bs):
                if self._stop_event.is_set():
                    break
                end = min(start + bs, n_total)
                chunk = data[:, start:end]
                n = self._push_slice(chunk, fs)

                with self._stats_lock:
                    self._samples_pushed += n
                    self._batch_count += 1
                    batch_count = self._batch_count
                    total = self._samples_pushed

                if (
                    not cfg.quiet
                    and cfg.stats_every_n_batches > 0
                    and batch_count % cfg.stats_every_n_batches == 0
                ):
                    print(f"[统计] 已累计推送约 {total} 个 EEG 样本")

            time.sleep(cfg.loop_sleep_sec)
