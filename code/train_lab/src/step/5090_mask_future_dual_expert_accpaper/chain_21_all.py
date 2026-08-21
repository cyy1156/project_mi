"""方案 21 链：F_mi_a → F_mi_080 → A2_pt → J1_tok。

用法：
  python chain_21_all.py
  python chain_21_all.py --from A2_pt
  python chain_21_all.py --max-folds 1
  python chain_21_all.py --with-a1-800
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms_registry import SERIES_21_OPTIONAL, SERIES_21_ORDER, assert_21_arm_flags  # noqa: E402


def main() -> None:
    assert_21_arm_flags()
    p = argparse.ArgumentParser(description="Scheme-21 LeJEPA alignment chain")
    p.add_argument("--from", dest="from_arm", default="", help="断点续跑，如 A2_pt")
    p.add_argument("--max-folds", type=int, default=0, help="0=五折；1=fold0")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--with-a1-800", action="store_true", help="F_mi_080 后附跑 A1_800")
    p.add_argument("--skip-fmi080", action="store_true")
    p.add_argument("--skip-j1", action="store_true")
    args = p.parse_args()

    order = list(SERIES_21_ORDER)
    if args.skip_fmi080:
        order = [a for a in order if a != "F_mi_080"]
    if args.skip_j1:
        order = [a for a in order if a != "J1_tok"]
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
            str(args.num_workers),
        ]
        print(f"[chain_21] RUN {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(HERE))
        if rc != 0:
            print(f"[chain_21] FAIL {arm} exit={rc}", flush=True)
            sys.exit(rc)
        if args.with_a1_800 and arm == "F_mi_080":
            cmd2 = cmd.copy()
            cmd2[cmd2.index("F_mi_080")] = "A1_800"
            print(f"[chain_21] RUN {' '.join(cmd2)}", flush=True)
            rc2 = subprocess.call(cmd2, cwd=str(HERE))
            if rc2 != 0:
                sys.exit(rc2)
    print("[chain_21] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
