"""方案 26 · 补 dump T-shallow 等 anchor 成员 softmax。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG24 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
PY = sys.executable


def dump_three(run_dir: Path, *, stage: str = "three") -> None:
    run_dir = Path(run_dir)
    cmd = [
        PY,
        str(PKG24 / "dump_probs.py"),
        "--run-dir",
        str(run_dir),
        "--stage",
        stage,
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PKG24))


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme26 dump anchor member probs")
    p.add_argument("--run-dir", type=Path, required=True, help=".../three")
    p.add_argument("--stage", default="three", choices=("three", "task"))
    args = p.parse_args()
    dump_three(args.run_dir, stage=args.stage)


if __name__ == "__main__":
    main()
