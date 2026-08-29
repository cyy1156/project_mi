#!/usr/bin/env python3
"""
一键启动操作台：HTTP + WebSocket 常驻，浏览器打开 Setup。

用法（仓库根 MI）:

  python -m experiment_game.tools.open_operator
  python -m experiment_game.tools.open_operator --no-browser
  python -m experiment_game.tools.open_operator --host 0.0.0.0   # 局域网监控端
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_CODE_ROOT = _REPO_ROOT / "code"
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from experiment_game.experiment.orchestrator import OperatorService


def _wire_implementations(svc: OperatorService) -> None:
    """入口层负责注入 offline/tools 具体实现（依赖倒置，见重构实施方案 §3.3）。

    OperatorService 未注入时也有惰性回退，此处显式接线以保证依赖方向单向。
    """
    from experiment_game.offline.phase4_service import run_phase4_for_session
    from experiment_game.offline.phase4_v2 import run as run_p4_cal
    from experiment_game.offline.phase4_v2_game import run as run_p4_game
    from experiment_game.tools.ft_subject_from_v3 import run_subject_finetune

    svc._phase4_runner = run_phase4_for_session  # noqa: SLF001
    svc._phase4_v2_pair_runner = (run_p4_cal, run_p4_game)  # noqa: SLF001
    svc._ft_runner = run_subject_finetune  # noqa: SLF001


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="操作者采集控制台")
    p.add_argument("--http-port", type=int, default=8080)
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="绑定地址：本机 127.0.0.1；局域网监控传 0.0.0.0（或具体 LAN IP）",
    )
    p.add_argument("--open-browser", action="store_true", default=True)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    svc = OperatorService(
        http_port=args.http_port,
        ws_port=args.ws_port,
        serve_host=args.host,
    )
    _wire_implementations(svc)
    open_browser = args.open_browser and not args.no_browser

    print("=== 操作者采集控制台 ===")
    print("关闭本窗口即结束服务。")
    print("默认：采集开 + 合成板；真机请在 Setup 选 Cyton，串口填设备管理器中的 COM（当前机常见 COM3）。")
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"远程模式：--host {args.host}（监控端用下方打印的局域网 URL）")

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
