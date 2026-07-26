#!/usr/bin/env python3
"""
一键启动诱导页：起 HTTP + WebSocket +（默认）Phase2 流程，并打开浏览器。

用法（仓库根或本目录均可）:

  python -m experiment_game.tools.open_induction
  python -m experiment_game.tools.open_induction --with-acq --acquire-trials 4

默认不加脑电（--no-acq），方便先看画面；需要采数时加 --with-acq。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_game.tools.run_phase2_session import main as phase2_main


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # 默认：跳过确认、打开浏览器、不采 EEG
    forwarded: list[str] = ["--yes"]
    with_acq = False
    rest: list[str] = []
    i = 0
    while i < len(raw):
        a = raw[i]
        if a == "--with-acq":
            with_acq = True
        elif a == "--no-acq":
            with_acq = False
        else:
            rest.append(a)
        i += 1

    if not with_acq:
        forwarded.append("--no-acq")
    forwarded.extend(rest)

    # 避免用户再传 --no-browser
    if "--no-browser" not in forwarded:
        forwarded.append("--open-browser")

    print("启动诱导页（将自动打开浏览器）…")
    print("关闭本窗口即结束会话。")
    return phase2_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
