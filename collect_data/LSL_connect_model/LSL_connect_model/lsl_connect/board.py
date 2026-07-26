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
@dataclass
class BoardConfig:
    serial_port: str = "COM10"
    use_synthetic: bool = True  # 是否使用合成板（无硬件测试）
    cyton_eeg_count: int = 8
    stream_buffer_size: int = 45000
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

    def __init__(self,config:Optional[BoardConfig]=None,serial_port:str="COM10",use_synthetic:bool=True) :
        """
                如果有真实的硬件把第37行换成下面两个，串口要是自己的串口号，如COM10
                    CytonBoard("COM10")
                    CytonBoard(BoardConfig(use_synthetic=True))
         """
        if config is None:
            config =BoardConfig(serial_port=serial_port)
        self.config = config
        self._board: Optional[BoardShim] = None
        self._stream_running = False
        self._board_id= (
            BoardIds.SYNTHETIC_BOARD.value
            if config.use_synthetic
            else BoardIds.CYTON_BOARD.value
        )

    @property  #装饰器把方法变成只读
    def is_connected(self) -> bool:
        return self._board is not None

    @property
    def board_id(self) -> int:
        return self._board_id

    @staticmethod
    def force_release_all(*, settle_sec: float = 0.0) -> None:
        """强制释放 BrainFlow 全局会话（Windows Cyton 二次 open 前常用）。"""
        try:
            BoardShim.release_all_sessions()
        except Exception as exc:
            print(f"[警告] release_all_sessions: {exc}")
        if settle_sec > 0:
            time.sleep(settle_sec)

    def connect(self, *, max_retries: int = 3) -> BoardShim:
        """连接板卡并 start_stream；真机失败时会重试 prepare_session。"""
        if self._board is not None:
            raise RuntimeError("板卡已连接，请先 disconnect()")

        real = not self.config.use_synthetic
        pre_delay = 0.35 if real else 0.0
        CytonBoard.force_release_all(settle_sec=pre_delay)

        params = BrainFlowInputParams()
        if real:
            params.serial_port = self.config.serial_port

        last_exc: Optional[Exception] = None
        attempts = max(1, int(max_retries)) if real else 1

        for attempt in range(attempts):
            if attempt > 0:
                wait = 0.6 + 0.5 * attempt
                print(
                    f"[提示] 串口重连第 {attempt + 1}/{attempts} 次，"
                    f"等待 {wait:.1f}s 后重试..."
                )
                CytonBoard.force_release_all(settle_sec=wait)

            self._board = BoardShim(self._board_id, params)
            try:
                self._board.prepare_session()
                self._board.start_stream(self.config.stream_buffer_size)
                self._stream_running = True
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                self._board = None
                CytonBoard.force_release_all(
                    settle_sec=0.25 if real else 0.0,
                )

        if last_exc is not None:
            raise last_exc

        if self.config.gui_streaming_enabled:
            url = (
                f"streaming_board://{self.config.gui_stream_ip}:"
                f"{self.config.gui_stream_port}"
            )
            self._board.add_streamer(url, BrainFlowPresets.DEFAULT_PRESET)
            print(f"[OK] GUI STREAMING 推流: {url}")
            if self.config.use_synthetic:
                print("[提示] GUI 里 BOARD 请选 Synthetic（合成板），不要选 Cyton")
            else:
                print("[提示] GUI 里 BOARD 请选 Cyton")

        if self.config.use_synthetic:
            print("[OK] 已启动 BrainFlow 合成板（无硬件测试模式）")
        else:
            print(f"[OK] 已连接 OpenBCI Cyton，串口: {self.config.serial_port}")
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

    def get_channel_indices(self) -> Tuple[np.ndarray, np.ndarray.int]:
        """
                返回 (eeg_channels, accel_channels, timestamp_channel)。
                合成板模式下 eeg 只保留前 cyton_eeg_count 路。
        """
        eeg =BoardShim.get_eeg_channels(self._board_id)
        accel =BoardShim.get_accel_channels(self._board_id)
        ts = BoardShim.get_timestamp_channel(self._board_id)

        if self.config.use_synthetic and self.config.cyton_eeg_count>0:
            eeg=eeg[ :self.config.cyton_eeg_count]

        return eeg, accel, ts

    def fetch_batch(self,num_sample :int)->np.ndarray:
        """拉取最近 num_samples 个采样（需已 connect）。"""
        return self.get_board_shim().get_current_board_data(num_sample)


    def fetch_new_batch(self) -> np.ndarray:
        """自上次调用以来新到的数据（不重叠、不截断）。"""
        return self.get_board_shim().get_board_data()
