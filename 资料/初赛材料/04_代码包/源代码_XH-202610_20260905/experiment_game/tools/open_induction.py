#!/usr/bin/env python3
"""
打开「被试诱导页」（仅浏览，不启动实验）。

正确用法：
  1. 先运行 open_operator.bat（操作台，提供 HTTP:8080 + WS:8765）
  2. 再运行本脚本，或直接打开 http://127.0.0.1:8080/

本脚本 **不再** 启动 Phase2 / v1 会话（旧行为会与操作台 v2/v3/v4 抢端口并显示错画面）。

若需要旧版「一键 Phase2 演示」，请显式：
  python -m experiment_game.tools.run_phase2_session --yes --no-acq --open-browser
"""

from __future__ import annotations

import argparse
import socket
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 0) or 0) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="打开被试诱导页（需操作台已运行；不启动 Phase2）"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=8080)
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument(
        "--legacy-phase2",
        action="store_true",
        help="兼容旧行为：启动 Phase2 会话（不推荐；会与操作台冲突）",
    )
    args = p.parse_args(argv)

    if args.legacy_phase2:
        print("警告：--legacy-phase2 将启动 v1 Phase2，可能与 open_operator 冲突。")
        from experiment_game.tools.run_phase2_session import main as phase2_main

        return phase2_main(["--yes", "--no-acq", "--open-browser"])

    subject = f"http://{args.host}:{args.http_port}/"
    operator = f"http://{args.host}:{args.http_port}/operator.html#setup"

    http_up = _port_open(args.host, args.http_port) and _http_ok(subject)
    ws_up = _port_open(args.host, args.ws_port)

    if not http_up:
        print("未检测到操作台 HTTP 服务。")
        print()
        print("请先双击运行：")
        print("  experiment_game\\open_operator.bat")
        print("（或 machines\\5070_laptop\\open_operator.bat）")
        print()
        print("操作台打开后：")
        print(f"  操作者：{operator}")
        print(f"  被试页：{subject}")
        print()
        print("本脚本只负责打开被试页，不会单独起实验流程。")
        return 2

    if not ws_up:
        print(f"警告：HTTP 已开，但 WebSocket :{args.ws_port} 未监听。")
        print("请确认用的是 open_operator.bat，而不是其它仅静态页进程。")

    print("=== 被试诱导页 ===")
    print(f"打开：{subject}")
    print("请保持操作台黑窗口不要关。选 v2/v3/v4 后在操作台点「开始」。")
    if not args.no_browser:
        try:
            webbrowser.open(subject)
        except Exception as exc:  # noqa: BLE001
            print(f"打开浏览器失败: {exc}")
            print(f"请手动打开: {subject}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
