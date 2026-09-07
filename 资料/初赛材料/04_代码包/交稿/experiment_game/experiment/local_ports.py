"""本机端口占用检测（Windows / 通用）。"""

from __future__ import annotations

import socket
import subprocess
import sys
from typing import Iterable, List, Optional, Tuple


def is_port_listening(host: str, port: int, *, timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _local_port(addr: str) -> Optional[int]:
    """从 netstat 本地地址解析端口，如 127.0.0.1:8765 / [::1]:8765 / *:8765。"""
    if addr.startswith("["):
        # [::1]:8765
        try:
            return int(addr.rsplit("]:", 1)[1])
        except (IndexError, ValueError):
            return None
    try:
        return int(addr.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def listening_pid(port: int) -> Optional[int]:
    """返回监听该端口的 PID；无法解析时返回 None。"""
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            ["netstat", "-ano"],
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        state = parts[3].upper()
        if state not in ("LISTENING", "LISTEN"):
            continue
        if _local_port(parts[1]) != port:
            continue
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None


def format_port_conflict(
    occupied: Iterable[Tuple[str, int, Optional[int]]],
    *,
    operator_url: str = "http://127.0.0.1:8080/operator.html#setup",
) -> str:
    lines: List[str] = ["端口已被占用，无法启动操作台。"]
    pids: List[int] = []
    for host, port, pid in occupied:
        if pid is not None:
            lines.append(f"  - {host}:{port}  占用 PID={pid}")
            pids.append(pid)
        else:
            lines.append(f"  - {host}:{port}  已被占用（未能解析 PID）")
    lines.append(f"若旧操作台窗口还在，请直接打开: {operator_url}")
    uniq = sorted(set(pids))
    if uniq:
        pid_args = " ".join(str(p) for p in uniq)
        lines.append("若要重启，请先结束占用进程，例如:")
        lines.append(f"  Stop-Process -Id {pid_args} -Force")
    lines.append("或换端口启动:")
    lines.append(
        "  python -m experiment_game.tools.open_operator "
        "--http-port 8081 --ws-port 8766"
    )
    return "\n".join(lines)


def check_ports_free(
    ports: Iterable[Tuple[str, int]],
) -> List[Tuple[str, int, Optional[int]]]:
    """返回已被占用的 (host, port, pid) 列表；空列表表示均可绑定。"""
    busy: List[Tuple[str, int, Optional[int]]] = []
    for host, port in ports:
        if is_port_listening(host, port):
            busy.append((host, port, listening_pid(port)))
    return busy
