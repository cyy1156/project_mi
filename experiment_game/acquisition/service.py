"""对 collect_data/lsl_connect ServiceManager 的薄封装。"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
        self._link_event_callback = None

    @property
    def manager(self):
        if self._mgr is None:
            raise RuntimeError("采集尚未 create；请先调用 create()")
        return self._mgr

    @property
    def serial_port(self) -> str:
        return self._serial_port

    def preflight_probe(self) -> Dict[str, Any]:
        """会话前快速探针：2s 内判断 L1/L2 无线链路，不启动 BrainFlow。"""
        if self._use_synthetic:
            return {
                "ok": True,
                "skipped": True,
                "serial_port": self._serial_port,
            }

        from lsl_connect.cyton_link import format_link_failure, probe_cyton_version

        probe = probe_cyton_version(self._serial_port)
        if probe.ok:
            return {
                "ok": True,
                "skipped": False,
                "serial_port": self._serial_port,
                "firmware_line": probe.firmware_line,
                "failure_kind": probe.failure_kind.value,
            }

        guidance = format_link_failure(
            probe.failure_kind,
            self._serial_port,
            probe=probe,
        )
        return {
            "ok": False,
            "skipped": False,
            "serial_port": self._serial_port,
            "failure_kind": probe.failure_kind.value,
            "firmware_line": probe.firmware_line,
            "guidance": guidance,
        }

    def create(self, on_link_event=None):
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
        self._link_event_callback = on_link_event
        self._mgr = ServiceManager(cfg, link_event_callback=on_link_event)
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

    def health_check(
        self,
        *,
        wait_s: float = 2.0,
        min_samples: int = 150,
        lsl_stream_name: str = "OpenBCI_EEG",
        resolve_lsl: bool = True,
        warmup_s: float = 1.0,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """启动后断言采集 RUNNING、样本推送增长、LSL 流可 resolve。

        Cyton 真机常需 1–2s 预热；串口已开但板子未推数时会重试若干次。
        """
        from lsl_connect.state import ServiceState

        mgr = self.manager
        st = mgr.get_status()
        state = st.get("state")
        if state != ServiceState.RUNNING.value:
            hint = ""
            if not self._use_synthetic:
                hint = (
                    f"；当前串口={self._serial_port}。"
                    "请关闭 OpenBCI GUI 串口直播后重试"
                )
            raise RuntimeError(
                f"采集未进入 RUNNING（state={state}）{hint}"
            )

        if warmup_s > 0:
            time.sleep(float(warmup_s))

        delta = 0
        s1 = int(st.get("samples_pushed") or 0)
        attempts = max(1, int(retries))
        for i in range(attempts):
            st0 = mgr.get_status()
            if st0.get("state") != ServiceState.RUNNING.value:
                raise RuntimeError(
                    f"采集在健康检查中退出 RUNNING（state={st0.get('state')}）；"
                    f"串口={self._serial_port}。请确认 Cyton 已开机、dongle 配对，"
                    "并关闭 OpenBCI GUI Serial 直播后重试"
                )
            worker_alive = bool(st0.get("worker_running"))
            s0 = int(st0.get("samples_pushed") or 0)
            time.sleep(float(wait_s))
            st2 = mgr.get_status()
            s1 = int(st2.get("samples_pushed") or 0)
            delta = s1 - s0
            if delta >= min_samples:
                break
            if i + 1 < attempts:
                print(
                    f"[operator] 样本仍不足（+{delta}/{min_samples}），"
                    f"第 {i + 2}/{attempts} 次重试…"
                    f"{'' if worker_alive else '（采集线程未运行）'}"
                )

        if delta < min_samples:
            hint = ""
            if not self._use_synthetic:
                hint = (
                    f"\n真机排查（串口 {self._serial_port} 往往已打开，但板卡未推数）：\n"
                    "  1) Cyton 主板开关打开、电池有电，LED 正常闪\n"
                    "  2) USB dongle 插牢；与板卡距离近、少遮挡\n"
                    "  3) 关闭 OpenBCI GUI 的 CYTON Serial 直播（会抢 COM/干扰）\n"
                    "  4) 设备管理器确认仍是该 COM；拔插 USB 后等 5s 再开会话\n"
                    "  5) 勿同时开两个操作台/采集进程"
                )
            raise RuntimeError(
                f"采集推送样本不足：累计 {attempts}×{wait_s:.1f}s 后增长 {delta}"
                f"（需 ≥ {min_samples}）{hint}"
            )

        lsl_ok = True
        lsl_detail = "skipped"
        if resolve_lsl:
            try:
                from pylsl import resolve_byprop

                streams = resolve_byprop("name", lsl_stream_name, timeout=3.0)
                lsl_ok = bool(streams)
                lsl_detail = f"resolved={len(streams)}"
                if not lsl_ok:
                    raise RuntimeError(
                        f"LSL 流 {lsl_stream_name} 未 resolve（timeout=3s）"
                    )
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"LSL resolve 失败: {exc}") from exc

        return {
            "state": state,
            "samples_pushed": s1,
            "delta_samples": delta,
            "wait_s": wait_s,
            "warmup_s": warmup_s,
            "retries": attempts,
            "lsl_stream": lsl_stream_name,
            "lsl_ok": lsl_ok,
            "lsl_detail": lsl_detail,
        }

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
