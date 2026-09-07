"""真机/合成板：RingBuffer + EEGBus + CsvRecorder 单写链路（总册 W2）。

替代 lsl_connect Recorder 双轨：板卡只推 LSL，本模块单一 Inlet 扇出推理与 eeg.csv。
归属 experiment/（依赖 RingBuffer），不放 runtime/ 以免反向依赖。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from experiment_game.core.channel_layout import DEVICE_CHANNEL_LABELS
from experiment_game.experiment.inference_v2 import RingBuffer
from experiment_game.runtime.csv_recorder import CsvRecorderSubscriber
from experiment_game.runtime.eeg_health import ensure_session_bus

_LOG = logging.getLogger(__name__)


class LiveEegCapture:
    """挂 LSL → Bus → CSV；对外暴露 ``buf`` 供会话推理复用。"""

    def __init__(
        self,
        eeg_csv_path: Union[str, Path],
        *,
        channel_labels: Optional[Sequence[str]] = None,
        stream_name: str = "OpenBCI_EEG",
        sample_rate_hz: float = 250.0,
        use_synthetic: bool = False,
        serial_port: str = "",
    ) -> None:
        self.eeg_csv_path = Path(eeg_csv_path)
        self.labels: List[str] = list(channel_labels or DEVICE_CHANNEL_LABELS)
        self.stream_name = str(stream_name)
        self.sample_rate_hz = float(sample_rate_hz)
        self.use_synthetic = bool(use_synthetic)
        self.serial_port = str(serial_port or "")
        self.buf = RingBuffer()
        self.csv = CsvRecorderSubscriber(self.eeg_csv_path, channel_labels=self.labels)
        self._started = False

    def start(self, *, lsl_timeout_s: float = 8.0) -> RingBuffer:
        bus = ensure_session_bus(self.buf)
        self.csv.open()
        bus.subscribe(self.csv)
        self.buf.attach_lsl(self.stream_name, timeout_s=float(lsl_timeout_s))
        self._started = True
        return self.buf

    def stop(self) -> Dict[str, Any]:
        """停 CSV + 关 Inlet；返回可写入 session 的 quality/meta 摘要。"""
        meta: Dict[str, Any] = {}
        bus = getattr(self.buf, "_bus", None)
        if bus is not None:
            try:
                bus.unsubscribe(self.csv)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("EEGBus 退订 CSV 失败: %r", exc)
        try:
            meta = self.csv.write_meta(
                sample_rate_hz=self.sample_rate_hz,
                use_synthetic=self.use_synthetic,
                serial_port=self.serial_port,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("写 eeg.meta 失败，仅回退 rows_written: %r", exc)
            meta = {"samples_written": int(self.csv.rows_written)}
        close_err: Optional[BaseException] = None
        try:
            self.csv.close()
        except Exception as exc:  # noqa: BLE001
            close_err = exc
            _LOG.error("eeg.csv close 失败（落盘可能不完整）rows=%s: %r", self.csv.rows_written, exc)
        try:
            self.buf.close()
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("RingBuffer close 失败: %r", exc)
        self._started = False
        if close_err is not None:
            raise RuntimeError(
                f"eeg.csv 关闭失败，落盘可能不完整（rows={self.csv.rows_written}）"
            ) from close_err
        return meta
