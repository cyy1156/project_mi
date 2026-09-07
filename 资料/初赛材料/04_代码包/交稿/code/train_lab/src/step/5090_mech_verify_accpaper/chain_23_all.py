"""方案 23 链：Tier1 O2s_m → … → A1_all；Tier2 条件触发。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms_registry import TIER1_ORDER, TIER2_ORDER, assert_23_arm_flags  # noqa: E402


def main() -> None:
    assert_23_arm_flags()
    p = argparse.ArgumentParser(description="Scheme-23 chain")
    p.add_argument("--from", dest="from_arm", default="", help="断点臂名，如 L025")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--tier2", action="store_true", help="跑 Tier2 P1_local/E1/E2")
    p.add_argument("--skip-calibration", action="store_true", help="跳过 O2s_m（已校）")
    args = p.parse_args()

    order = list(TIER2_ORDER if args.tier2 else TIER1_ORDER)
    if args.skip_calibration and not args.tier2:
        order = [a for a in order if a != "O2s_m"]
    if args.from_arm:
        order = order[order.index(args.from_arm) :]

    py = sys.executable
    for arm in order:
        cmd = [
            py,
            str(HERE / "run_arm.py"),
            "--arm",
            arm,
            "--max-folds",
            str(args.max_folds),
            "--num-workers",
            "0",
        ]
        print(f"[chain_23] RUN {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(HERE))
        if rc != 0:
            print(f"[chain_23] FAIL {arm} exit={rc}", flush=True)
            sys.exit(rc)
    print("[chain_23] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
