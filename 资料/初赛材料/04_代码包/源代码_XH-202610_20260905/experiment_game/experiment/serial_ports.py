"""枚举本机串口（Windows 优先；无 pyserial 依赖）。"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, List


def list_serial_ports() -> List[Dict[str, str]]:
    """返回 [{device, description}, ...]，按 COM 号排序。"""
    ports: List[Dict[str, str]] = []
    if sys.platform == "win32":
        ports = _list_windows_registry()
        if not ports:
            ports = _list_windows_powershell()
    else:
        ports = _list_posix_dev()
    return _sort_ports(ports)


def _sort_ports(ports: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def key(p: Dict[str, str]) -> tuple:
        m = re.search(r"(\d+)", p.get("device") or "")
        return (int(m.group(1)) if m else 9999, p.get("device") or "")

    seen = set()
    out = []
    for p in sorted(ports, key=key):
        d = p.get("device") or ""
        if not d or d in seen:
            continue
        seen.add(d)
        out.append({"device": d, "description": p.get("description") or d})
    return out


def _list_windows_registry() -> List[Dict[str, str]]:
    try:
        import winreg  # type: ignore
    except ImportError:
        return []
    out: List[Dict[str, str]] = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
    except OSError:
        return []
    try:
        i = 0
        while True:
            try:
                name, value, _ = winreg.EnumValue(key, i)
                out.append({"device": str(value), "description": str(name)})
                i += 1
            except OSError:
                break
    finally:
        winreg.CloseKey(key)
    return out


def _list_windows_powershell() -> List[Dict[str, str]]:
    import subprocess

    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_SerialPort | "
                "Select-Object DeviceID,Name | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    raw = (proc.stdout or "").strip()
    if not raw:
        return []
    import json

    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    out: List[Dict[str, str]] = []
    for row in data or []:
        if not isinstance(row, dict):
            continue
        dev = str(row.get("DeviceID") or "").strip()
        name = str(row.get("Name") or dev).strip()
        if dev:
            out.append({"device": dev, "description": name})
    return out


def _list_posix_dev() -> List[Dict[str, str]]:
    from pathlib import Path

    out: List[Dict[str, str]] = []
    base = Path("/dev")
    if not base.is_dir():
        return out
    for p in sorted(base.iterdir()):
        name = p.name
        if name.startswith(("ttyUSB", "ttyACM")) or name.startswith("cu."):
            out.append({"device": str(p), "description": name})
    return out
