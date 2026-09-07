"""
第 4 课：BrainFlow 板卡封装。
把 prepare_session / start_stream / stop / release 收到类里，供后续 acquisition_worker 复用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

from lsl_connect.cyton_link import (
    ConnectAttemptLog,
    CytonConnectReport,
    CytonLinkProbe,
    LinkFailureKind,
    classify_brainflow_error,
    ensure_serial_idle,
    format_link_failure,
    probe_cyton_version,
)


@dataclass
class BoardConfig:
    serial_port: str = "COM10"
    use_synthetic: bool = True  # 是否使用合成板（无硬件测试）
    cyton_eeg_count: int = 8
    stream_buffer_size: int = 45000
    # BrainFlow / 串口连接（真机）
    serial_timeout_sec: int = 15
    connect_max_retries: int = 3
    pre_connect_probe: bool = True  # connect 前 probe_cyton_version
    ensure_idle_before_connect: bool = True  # s 停裸串口遗留流
    post_connect_min_samples: int = 50  # 连上后缓冲区内最少样本数
    post_connect_warmup_sec: float = 0.8
    # OpenBCI GUI 7「STREAMING (from external)」旁路（BrainFlow UDP，与 LSL 并行）
    gui_streaming_enabled: bool = False
    gui_stream_ip: str = "225.1.1.1"
    gui_stream_port: int = 6677


class CytonBoard:
    """
      OpenBCI Cyton（或合成板）连接封装。
      用法:
          board = CytonBoard(BoardConfig(use_synthetic=True))
          board.connect()
          ...
          board.disconnect()
      """

    def __init__(
        self,
        config: Optional[BoardConfig] = None,
        serial_port: str = "COM10",
        use_synthetic: bool = True,
    ) -> None:
        if config is None:
            config = BoardConfig(serial_port=serial_port, use_synthetic=use_synthetic)
        self.config = config
        self._board: Optional[BoardShim] = None
        self._stream_running = False
        self._last_connect_report: Optional[CytonConnectReport] = None
        self._board_id = (
            BoardIds.SYNTHETIC_BOARD.value
            if config.use_synthetic
            else BoardIds.CYTON_BOARD.value
        )

    @property
    def is_connected(self) -> bool:
        return self._board is not None

    @property
    def board_id(self) -> int:
        return self._board_id

    @property
    def last_connect_report(self) -> Optional[CytonConnectReport]:
        return self._last_connect_report

    @staticmethod
    def force_release_all(*, settle_sec: float = 0.0) -> None:
        """强制释放 BrainFlow 全局会话（Windows Cyton 二次 open 前常用）。"""
        try:
            BoardShim.release_all_sessions()
        except Exception as exc:
            print(f"[警告] release_all_sessions: {exc}")
        if settle_sec > 0:
            time.sleep(settle_sec)

    @staticmethod
    def ensure_serial_idle(serial_port: str, *, settle_sec: float = 0.5) -> bool:
        """兼容旧调用；返回是否 probe 成功。"""
        probe = ensure_serial_idle(serial_port, settle_sec=settle_sec)
        return probe.ok

    def connect(self, *, max_retries: Optional[int] = None) -> BoardShim:
        """连接板卡并 start_stream；真机含串口预检、停流、BF 重试。"""
        if self._board is not None:
            raise RuntimeError("板卡已连接，请先 disconnect()")

        cfg = self.config
        real = not cfg.use_synthetic
        port = cfg.serial_port
        attempts_n = max(1, int(max_retries or cfg.connect_max_retries)) if real else 1

        report = CytonConnectReport(port=port, ok=False)
        self._last_connect_report = report

        if not real:
            return self._connect_synthetic()

        pre_delay = 0.35
        CytonBoard.force_release_all(settle_sec=pre_delay)

        params = BrainFlowInputParams()
        params.serial_port = port
        params.timeout = max(1, int(cfg.serial_timeout_sec))

        last_exc: Optional[Exception] = None
        last_kind = LinkFailureKind.UNKNOWN
        last_probe: Optional[CytonLinkProbe] = None

        for attempt in range(attempts_n):
            log = ConnectAttemptLog(attempt=attempt + 1)

            if attempt > 0:
                wait = 0.6 + 0.5 * attempt
                print(
                    f"[提示] 串口重连第 {attempt + 1}/{attempts_n} 次，"
                    f"等待 {wait:.1f}s 后重试..."
                )
                CytonBoard.force_release_all(settle_sec=wait)

            if cfg.pre_connect_probe:
                last_probe = probe_cyton_version(port)
                log.probe = last_probe
                print(f"[链路] {last_probe.summary()}")
                if not last_probe.ok:
                    last_kind = last_probe.failure_kind
                    last_exc = RuntimeError(last_probe.summary())
                    log.error = last_probe.summary()
                    report.attempts.append(log)
                    continue

            if cfg.ensure_idle_before_connect:
                idle_probe = ensure_serial_idle(port)
                log.probe = idle_probe
                last_probe = idle_probe
                if not idle_probe.ok:
                    print(f"[链路] 停流预检: {idle_probe.summary()}")
                    last_kind = idle_probe.failure_kind
                    last_exc = RuntimeError(idle_probe.summary())
                    log.error = idle_probe.summary()
                    report.attempts.append(log)
                    continue

            CytonBoard.force_release_all(settle_sec=0.25)

            self._board = BoardShim(self._board_id, params)
            try:
                self._board.prepare_session()
                self._board.start_stream(cfg.stream_buffer_size)
                self._stream_running = True

                time.sleep(float(cfg.post_connect_warmup_sec))
                warm_n = self._board.get_current_board_data(
                    max(cfg.post_connect_min_samples * 2, 125),
                ).shape[1]
                log.samples_after_connect = warm_n
                if warm_n < cfg.post_connect_min_samples:
                    raise RuntimeError(
                        f"连接后样本不足 ({warm_n} < {cfg.post_connect_min_samples})"
                    )

                log.brainflow_ok = True
                report.attempts.append(log)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                last_kind = classify_brainflow_error(exc)
                if "样本不足" in str(exc):
                    last_kind = LinkFailureKind.STREAM_STALL
                log.error = str(exc)
                report.attempts.append(log)
                self._board = None
                self._stream_running = False
                CytonBoard.force_release_all(settle_sec=0.25)

        if last_exc is not None:
            report.failure_kind = last_kind
            report.message = format_link_failure(
                last_kind,
                port,
                probe=last_probe,
                brainflow_msg=str(last_exc),
            )
            raise RuntimeError(report.message) from last_exc

        report.ok = True
        report.message = last_probe.summary() if last_probe else f"{port} 已连接"

        if cfg.gui_streaming_enabled:
            url = (
                f"streaming_board://{cfg.gui_stream_ip}:"
                f"{cfg.gui_stream_port}"
            )
            assert self._board is not None
            self._board.add_streamer(url, BrainFlowPresets.DEFAULT_PRESET)
            print(f"[OK] GUI STREAMING 推流: {url}")

        print(f"[OK] 已连接 OpenBCI Cyton，串口: {port}")
        assert self._board is not None
        return self._board

    def _connect_synthetic(self) -> BoardShim:
        CytonBoard.force_release_all(settle_sec=0.0)
        params = BrainFlowInputParams()
        self._board = BoardShim(self._board_id, params)
        self._board.prepare_session()
        self._board.start_stream(self.config.stream_buffer_size)
        self._stream_running = True
        print("[OK] 已启动 BrainFlow 合成板（无硬件测试模式）")
        report = CytonConnectReport(
            port="synthetic",
            ok=True,
            message="合成板",
        )
        self._last_connect_report = report
        return self._board

    def stop_stream_only(self) -> None:
        """仅停流，用于采集线程 join 前先打断 get_board_data 阻塞。"""
        if self._board is None or not self._stream_running:
            return
        try:
            self._board.stop_stream()
            self._stream_running = False
        except Exception as exc:
            print(f"[警告] stop_stream: {exc}")
            self._stream_running = False

    def disconnect(self) -> None:
        """停止推流并释放会话。"""
        if self._board is None:
            return

        board = self._board
        self._board = None
        real = not self.config.use_synthetic

        if self._stream_running:
            try:
                board.stop_stream()
            except Exception as exc:
                print(f"[警告] stop_stream: {exc}")
            self._stream_running = False
            if real:
                time.sleep(0.15)

        try:
            board.release_session()
        except Exception as exc:
            print(f"[警告] release_session: {exc}")

        CytonBoard.force_release_all(settle_sec=0.25 if real else 0.0)
        print("[OK] 已释放硬件资源")

    def get_board_shim(self) -> BoardShim:
        if self._board is None:
            raise RuntimeError("板卡未连接，请先 connect()")
        return self._board

    def get_channel_indices(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        返回 (eeg_channels, accel_channels, timestamp_channel)。
        合成板模式下 eeg 只保留前 cyton_eeg_count 路。
        """
        eeg = BoardShim.get_eeg_channels(self._board_id)
        accel = BoardShim.get_accel_channels(self._board_id)
        ts = BoardShim.get_timestamp_channel(self._board_id)

        if self.config.use_synthetic and self.config.cyton_eeg_count > 0:
            eeg = eeg[: self.config.cyton_eeg_count]

        return eeg, accel, ts

    def fetch_batch(self, num_sample: int) -> np.ndarray:
        """拉取最近 num_samples 个采样（需已 connect）。"""
        return self.get_board_shim().get_current_board_data(num_sample)

    def fetch_new_batch(self) -> np.ndarray:
        """自上次调用以来新到的数据（不重叠、不截断）。"""
        return self.get_board_shim().get_board_data()
