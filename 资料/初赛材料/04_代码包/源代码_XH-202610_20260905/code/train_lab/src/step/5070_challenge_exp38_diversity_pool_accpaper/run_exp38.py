# -*- coding: utf-8 -*-
"""Exp38 编排。

用法：
  python run_exp38.py --stage d1
  python run_exp38.py --stage d2
  python run_exp38.py --stage all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_STEP = Path(__file__).resolve().parent


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(_STEP / script), *(extra or [])]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_STEP))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("d1", "d2", "all"), default="all")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()
    extra = []
    if args.max_folds:
        extra += ["--max-folds", str(args.max_folds)]
    if args.run_tag:
        extra += ["--run-tag", args.run_tag]

    if args.stage in ("d1", "all"):
        _run("train_classical.py", extra)
        _run("train_neural.py", extra)
    if args.stage in ("d2", "all"):
        d2 = []
        if args.run_tag:
            d2 += ["--run-tag", args.run_tag]
        _run("nested_greedy_select.py", d2)
    print("DONE", args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
