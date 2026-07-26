#!/usr/bin/env python3
"""
一键启动操作台：HTTP + WebSocket 常驻，浏览器打开 Setup。

用法（仓库根 MI）:

  python -m experiment_game.tools.open_operator
  python -m experiment_game.tools.open_operator --no-browser
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_game.experiment.orchestrator import OperatorService


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="操作者采集控制台")
    p.add_argument("--http-port", type=int, default=8080)
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--open-browser", action="store_true", default=True)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    svc = OperatorService(http_port=args.http_port, ws_port=args.ws_port)
    open_browser = args.open_browser and not args.no_browser

    print("=== 操作者采集控制台 ===")
    print("关闭本窗口即结束服务。")
    print("默认：采集开 + 合成板；真机请在 Setup 选 Cyton，串口 COM5。")

    try:
        svc.start()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if open_browser:
        try:
            webbrowser.open(svc.operator_url)
        except Exception as exc:  # noqa: BLE001
            print(f"打开浏览器失败: {exc}", file=sys.stderr)
            print(f"请手动打开: {svc.operator_url}")

    try:
        while True:
            import time

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n用户中断", file=sys.stderr)
        return 130
    finally:
        svc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
