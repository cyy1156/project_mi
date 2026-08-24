"""
OpenBCI Cyton 串口 / 无线链路预检与状态恢复。

问题背景（三层故障，表象常被误报为「丢包」）
-------------------------------------------

L1  USB dongle ↔ PC（COM 口）
    - 现象：UNABLE_TO_OPEN_PORT；COM 在设备管理器消失
    - 原因：USB 松动、驱动、端口被 GUI/其它进程占用

L2  Cyton 主板 ↔ 2.4G dongle（无线）
    - 现象：COM 能 open，但 ``v`` 回 0 字节；或流中 Wrong end byte / 样本停滞
    - 原因：板子未开机、配对断、距离远、干扰 —— 用户常称「接触不良」

L3  OpenBCI 串口协议状态（idle vs streaming）
    - 现象：裸串口 ``v`` 正常，BrainFlow 报 ``welcome characters``
    - 原因：GUI / raw_probe 已发 ``b`` 开流，板卡不再回 welcome；需先发 ``s`` 停流

L4  BrainFlow 会话残留
    - 现象：上次 Python/GUI 未 release，二次 connect 失败
    - 处理：``BoardShim.release_all_sessions()`` + 等待

与「丢包率 drop_rate_pct」的区别
--------------------------------
- ``drop_rate_pct`` 是 LSL 推送 vs CSV 写入差，且含录制启动对齐；不是 UART 比特丢包。
- 连不上 / 流停滞属于 **链路层** 问题，应在 connect 前/中处理，而非事后看 drop_rate。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

_RE_FIRMWARE = re.compile(r"Firmware:\s*v[\d.]+", re.IGNORECASE)
_RE_OPENBCI = re.compile(r"OpenBCI[^\x00-\x1f\r\n]{0,80}", re.IGNORECASE)
_PRINTABLE_ASCII = re.compile(r"[^\x20-\x7E]")


class LinkFailureKind(str, Enum):
    """链路故障分类，供 UI / 日志使用。"""

    OK = "ok"
    PORT_BUSY = "port_busy"  # L1: COM 打不开
    NO_WIRELESS = "no_wireless"  # L2: dongle 在，Cyton 不应
    STREAMING_BUSY = "streaming_busy"  # L3: 板卡已在裸串口流模式
    BRAINFLOW_HANDSHAKE = "brainflow_handshake"  # L3/L4: BF prepare 失败
    STREAM_STALL = "stream_stall"  # L2: 连上后长时间无样本
    UNKNOWN = "unknown"


@dataclass
class CytonLinkProbe:
    """一次 ``v`` 预检结果。"""

    port: str
    ok: bool
    version_bytes: int = 0
    firmware_line: str = ""
    raw_head: bytes = b""
    error: str = ""
    failure_kind: LinkFailureKind = LinkFailureKind.OK

    def summary(self) -> str:
        if self.ok:
            fw = self.firmware_line or "(未知版本)"
            return f"{self.port} Cyton 在线 — {fw}"
        if self.failure_kind == LinkFailureKind.NO_WIRELESS:
            return (
                f"{self.port} dongle 已识别，但 Cyton 无回应（无线未配对或板子未开机）"
            )
        if self.failure_kind == LinkFailureKind.PORT_BUSY:
            return f"{self.port} 无法打开（端口占用或已拔出）"
        return f"{self.port} 预检失败: {self.error or '无 OpenBCI 回应'}"


@dataclass
class ConnectAttemptLog:
    """连接尝试记录，便于操作台展示。"""

    attempt: int
    probe: Optional[CytonLinkProbe] = None
    brainflow_ok: bool = False
    samples_after_connect: int = 0
    error: str = ""


@dataclass
class CytonConnectReport:
    """``CytonBoard.connect()`` 完整报告。"""

    port: str
    ok: bool
    attempts: list[ConnectAttemptLog] = field(default_factory=list)
    failure_kind: LinkFailureKind = LinkFailureKind.OK
    message: str = ""


def format_firmware_display(raw: Union[bytes, str]) -> str:
    """从 ``v`` 命令原始字节中提取可读的固件/型号行，避免二进制前缀乱码。"""
    if isinstance(raw, bytes):
        text = raw.decode("ascii", errors="ignore")
    else:
        text = str(raw or "")

    match = _RE_FIRMWARE.search(text)
    if match:
        return match.group(0).strip()

    match = _RE_OPENBCI.search(text)
    if match:
        cleaned = _PRINTABLE_ASCII.sub("", match.group(0)).strip()
        if cleaned:
            return cleaned[:64]

    for line in text.splitlines():
        cleaned = _PRINTABLE_ASCII.sub("", line).strip()
        if "OpenBCI" in cleaned or "Firmware" in cleaned:
            return cleaned[:64]

    for run in re.findall(r"[\x20-\x7E]{4,}", text):
        if "OpenBCI" in run or "Firmware" in run:
            return run[:64]

    return ""


def short_firmware_display(raw: Union[bytes, str]) -> str:
    """操作台链路面板用的短固件串。"""
    text = format_firmware_display(raw)
    if not text:
        return ""
    if "Firmware:" in text:
        version = text.split("Firmware:", 1)[-1].strip()
        return version[:32] if version else text[:48]
    return text[:48]


def _parse_firmware(raw: bytes) -> str:
    return format_firmware_display(raw)


def probe_cyton_version(
    serial_port: str,
    *,
    settle_sec: float = 0.3,
) -> CytonLinkProbe:
    """L2 预检：仅发 ``v``，确认无线链路是否通。"""
    try:
        import serial
    except ImportError:
        return CytonLinkProbe(
            port=serial_port,
            ok=True,
            firmware_line="(未安装 pyserial，跳过预检)",
        )

    try:
        ser = serial.Serial(
            serial_port,
            baudrate=115200,
            timeout=2.0,
            write_timeout=2.0,
        )
    except Exception as exc:  # noqa: BLE001
        return CytonLinkProbe(
            port=serial_port,
            ok=False,
            error=str(exc),
            failure_kind=LinkFailureKind.PORT_BUSY,
        )

    try:
        ser.dtr = True
        ser.rts = True
        time.sleep(settle_sec)
        ser.reset_input_buffer()
        ser.write(b"v")
        time.sleep(1.0)
        n = ser.in_waiting
        raw = ser.read(n) if n else b""
    finally:
        ser.close()

    if raw and b"OpenBCI" in raw:
        return CytonLinkProbe(
            port=serial_port,
            ok=True,
            version_bytes=len(raw),
            firmware_line=_parse_firmware(raw),
            raw_head=raw[:120],
        )

    return CytonLinkProbe(
        port=serial_port,
        ok=False,
        version_bytes=len(raw),
        raw_head=raw[:120],
        failure_kind=LinkFailureKind.NO_WIRELESS,
        error="v 命令无 OpenBCI 回应",
    )


def ensure_serial_idle(
    serial_port: str,
    *,
    settle_sec: float = 0.5,
) -> CytonLinkProbe:
    """L3 恢复：``s`` 停裸串口流 → ``v`` 确认 idle。返回最终 probe 结果。"""
    try:
        import serial
    except ImportError:
        return probe_cyton_version(serial_port, settle_sec=settle_sec)

    try:
        ser = serial.Serial(
            serial_port,
            baudrate=115200,
            timeout=2.0,
            write_timeout=2.0,
        )
    except Exception as exc:  # noqa: BLE001
        return CytonLinkProbe(
            port=serial_port,
            ok=False,
            error=str(exc),
            failure_kind=LinkFailureKind.PORT_BUSY,
        )

    try:
        ser.dtr = True
        ser.rts = True
        time.sleep(0.3)
        ser.reset_input_buffer()
        ser.write(b"s")
        time.sleep(0.6)
        ser.reset_input_buffer()
        ser.write(b"v")
        time.sleep(1.0)
        n = ser.in_waiting
        raw = ser.read(n) if n else b""
    finally:
        ser.close()
        time.sleep(settle_sec)

    if raw and b"OpenBCI" in raw:
        return CytonLinkProbe(
            port=serial_port,
            ok=True,
            version_bytes=len(raw),
            firmware_line=_parse_firmware(raw),
            raw_head=raw[:120],
        )

    # 有字节但非 OpenBCI → 可能仍在 binary 流模式
    kind = LinkFailureKind.STREAMING_BUSY if raw else LinkFailureKind.NO_WIRELESS
    return CytonLinkProbe(
        port=serial_port,
        ok=False,
        version_bytes=len(raw),
        raw_head=raw[:120],
        failure_kind=kind,
        error="停流后仍无有效 OpenBCI 版本串",
    )


def classify_brainflow_error(exc: Exception) -> LinkFailureKind:
    raw = str(exc)
    upper = raw.upper()
    if "UNABLE_TO_OPEN_PORT" in upper:
        return LinkFailureKind.PORT_BUSY
    if "WELCOME" in upper or "BOARD_NOT_READY" in upper:
        return LinkFailureKind.STREAMING_BUSY
    if "UNABLE TO PREPARE" in upper:
        return LinkFailureKind.BRAINFLOW_HANDSHAKE
    return LinkFailureKind.UNKNOWN


def format_link_failure(
    kind: LinkFailureKind,
    port: str,
    *,
    probe: Optional[CytonLinkProbe] = None,
    brainflow_msg: str = "",
) -> str:
    """生成可操作的中文提示（操作台 / CLI）。"""
    lines: list[str] = []
    if probe is not None and probe.firmware_line:
        lines.append(f"预检: {probe.summary()}")

    if kind == LinkFailureKind.PORT_BUSY:
        lines.append(
            f"无法打开串口 {port}。\n"
            "  ① 设备管理器确认 COM 号  ② 关闭 OpenBCI GUI\n"
            "  ③ 无其它 python/采集进程  ④ USB dongle 重新插拔"
        )
    elif kind == LinkFailureKind.NO_WIRELESS:
        lines.append(
            f"串口 {port} 可访问，但 Cyton 主板无回应（无线「接触不良」）。\n"
            "  ① Cyton 电源打开、LED 正常  ② dongle 与板卡距离 <1m\n"
            "  ③ 关开 Cyton 重新配对  ④ 换 USB 口插 dongle"
        )
    elif kind == LinkFailureKind.STREAMING_BUSY:
        lines.append(
            f"串口 {port} 上板卡处于裸串口推流状态，BrainFlow 无法握手。\n"
            "  ① 已自动尝试 s 停流；若仍失败请 Cyton 关机再开\n"
            "  ② 勿在 OpenBCI GUI Serial 直播与操作台同时采集\n"
            "  ③ 勿先跑 com3_raw_probe --stream 再立刻开操作台"
        )
    elif kind == LinkFailureKind.STREAM_STALL:
        lines.append(
            f"BrainFlow 已连接 {port}，但启动后样本不增长。\n"
            "  无线链路可能在推流中途断开 — 请检查 Cyton 电量与 2.4G 信号"
        )
    elif kind == LinkFailureKind.BRAINFLOW_HANDSHAKE:
        lines.append(
            f"BrainFlow 握手失败 ({port})。\n"
            "  常见：板卡未 idle、会话未释放、或无线瞬断"
        )
    else:
        lines.append(f"连接 {port} 失败: {brainflow_msg or kind.value}")

    if brainflow_msg and kind not in (
        LinkFailureKind.BRAINFLOW_HANDSHAKE,
        LinkFailureKind.UNKNOWN,
    ):
        lines.append(f"BrainFlow: {brainflow_msg}")
    return "\n".join(lines)
