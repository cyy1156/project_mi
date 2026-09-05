"""兼容旧入口：转发到 run_all.py / baseline_*.py。

推荐：
  python baseline_eegnet.py
  python run_all.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
ALL_MODELS = ("shallow", "deep", "conformer", "eegnet", "eegtcnet")


def main() -> None:
    p = argparse.ArgumentParser(
        description="兼容入口：请优先使用 baseline_*.py / run_all.py"
    )
    p.add_argument("--models", default=",".join(ALL_MODELS))
    args, unknown = p.parse_known_args()
    cmd = [
        sys.executable,
        str(DIR / "run_all.py"),
        "--models",
        args.models,
        *unknown,
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(DIR)))


if __name__ == "__main__":
    main()
