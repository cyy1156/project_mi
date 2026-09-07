"""Cyton ``v`` 响应固件串解析（防链路面板乱码）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[3]
    / "collect_data"
    / "LSL_connect_model"
    / "LSL_connect_model"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lsl_connect.cyton_link import (  # noqa: E402
    format_firmware_display,
    short_firmware_display,
)


def test_format_firmware_strips_binary_prefix() -> None:
    raw = b"\x00\xff\x00\x00OpenBCI V3 8-16 channel\r\n"
    assert format_firmware_display(raw) == "OpenBCI V3 8-16 channel"


def test_format_firmware_extracts_version_line() -> None:
    raw = b"\x01\x02Firmware: v3.1.5\r\nOpenBCI V3 8-16 channel"
    assert format_firmware_display(raw) == "Firmware: v3.1.5"


def test_short_firmware_prefers_version_suffix() -> None:
    raw = b"\xffOpenBCI V3 8-16 channel\nFirmware: v3.1.5"
    assert short_firmware_display(raw) == "v3.1.5"


def test_format_firmware_empty_on_noise() -> None:
    assert format_firmware_display(b"\x00\xff\x01\x02") == ""
