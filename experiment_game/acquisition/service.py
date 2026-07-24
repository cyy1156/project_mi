"""对 collect_data/lsl_connect ServiceManager 的薄封装。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

DEFAULT_CHANNEL_LABELS: List[str] = [
    "C3",
    "C4",
    "CZ",
    "CP3",
    "CP4",
    "CPZ",
    "FC3",
    "FC4",
]

LSL_CONNECT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "collect_data"
    / "LSL_connect_model"
    / "LSL_connect_model"
)


def ensure_lsl_connect_on_path(root: Optional[Path] = None) -> Path:
    path = Path(root) if root is not None else LSL_CONNECT_ROOT
    if not path.is_dir():
        raise FileNotFoundError(f"找不到 lsl_connect 工程根目录: {path}")
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return path


class AcquisitionFacade:
    """启停 BrainFlow→LSL 采集与 CSV 录制；不复制板卡逻辑。"""

    def __init__(
        self,
        *,
        use_synthetic: bool = True,
        serial_port: str = "COM3",
        channel_labels: Optional[Sequence[str]] = None,
        lsl_connect_root: Optional[Path] = None,
        filter_enabled: bool = True,
        bandpass_low_hz: float = 0.5,
        bandpass_high_hz: float = 45.0,
        notch_low_hz: float = 49.0,
        notch_high_hz: float = 51.0,
    ) -> None:
        self._root = ensure_lsl_connect_on_path(lsl_connect_root)
        self._use_synthetic = use_synthetic
        self._serial_port = serial_port
        self._labels = list(channel_labels or DEFAULT_CHANNEL_LABELS)
        self._filter_enabled = filter_enabled
        self._bandpass_low_hz = float(bandpass_low_hz)
        self._bandpass_high_hz = float(bandpass_high_hz)
        self._notch_low_hz = float(notch_low_hz)
        self._notch_high_hz = float(notch_high_hz)
        self._mgr = None

    @property
    def manager(self):
        if self._mgr is None:
            raise RuntimeError("采集尚未 create；请先调用 create()")
        return self._mgr

    def create(self):
        from lsl_connect.board import BoardConfig
        from lsl_connect.lsl_streams import LslStreamConfig
        from lsl_connect.preprocessing import PreprocessConfig
        from lsl_connect.acquisition_work import AcquisitionConfig
        from lsl_connect.recording_config import RecordingConfig
        from lsl_connect.service_manager import ServiceManager, ServiceManagerConfig

        n = len(self._labels)
        board = BoardConfig(
            use_synthetic=self._use_synthetic,
            serial_port=self._serial_port,
            cyton_eeg_count=n,
        )
        cfg = ServiceManagerConfig(
            board_config=board,
            lsl=LslStreamConfig(
                sample_rate=250,
                channel_count=n,
                use_synthetic=self._use_synthetic,
                eeg_labels=list(self._labels),
            ),
            preprocess=PreprocessConfig(
                sample_rate=250,
                filter_enabled=self._filter_enabled,
                bandpass_low_hz=self._bandpass_low_hz,
                bandpass_high_hz=self._bandpass_high_hz,
                notch_low_hz=self._notch_low_hz,
                notch_high_hz=self._notch_high_hz,
            ),
            acquisition=AcquisitionConfig(),
            recording=RecordingConfig(),
        )
        self._mgr = ServiceManager(cfg)
        ok, msg = self._mgr.set_eeg_channel_labels(",".join(self._labels))
        if not ok:
            raise RuntimeError(f"设置通道标签失败: {msg}")
        return self

    def start(self, eeg_csv_path: Path) -> None:
        mgr = self.manager
        try:
            ok = mgr.start_acquisition()
        except Exception as exc:  # noqa: BLE001
            hint = ""
            if not self._use_synthetic:
                hint = (
                    "；请确认：① 串口正确 ② 已关闭 OpenBCI GUI 串口直播 "
                    "③ USB 连接稳定"
                )
            raise RuntimeError(f"启动采集异常: {exc}{hint}") from exc
        if not ok:
            err = getattr(mgr, "_last_error", None) or mgr.get_status()
            hint = ""
            if not self._use_synthetic:
                hint = (
                    f"；当前串口={self._serial_port}。"
                    "请关闭 OpenBCI GUI 的 Serial/直播，核对设备管理器 COM 口后重试"
                )
            raise RuntimeError(f"启动采集失败: {err}{hint}")
        ok, msg = mgr.start_recording(str(eeg_csv_path))
        if not ok:
            mgr.stop_acquisition()
            raise RuntimeError(f"启动录制失败: {msg}")

    def stop(self) -> dict:
        mgr = self.manager
        report_dict = {}
        ok, msg, report = mgr.stop_recording()
        if report is not None:
            report_dict = report.to_dict() if hasattr(report, "to_dict") else {}
        mgr.stop_acquisition()
        return {"stop_recording_ok": ok, "message": msg, "quality": report_dict}

    def shutdown(self) -> None:
        if self._mgr is not None:
            self._mgr.shutdown()
            self._mgr = None
