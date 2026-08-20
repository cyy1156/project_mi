"""T 系列 v2 链：T1 → T1_aux → T1_128（5060/5090 同序）。

用法：
  python chain_t_all.py
  python chain_t_all.py --from T1_aux
  python chain_t_all.py --max-folds 1          # fold0 冒烟
  python chain_t_all.py --skip-t1-128          # 仅 T1 + T1_aux
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms_registry import T_SERIES_ORDER, assert_t_arm_flags  # noqa: E402


def main() -> None:
    assert_t_arm_flags()
    p = argparse.ArgumentParser(description="T-series chain (v2 Token+Phase Predictor)")
    p.add_argument("--from", dest="from_arm", default="", help="断点续跑，如 T1_aux")
    p.add_argument("--max-folds", type=int, default=0, help="0=五折；1=fold0")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--skip-t1-128", action="store_true")
    args = p.parse_args()

    order = list(T_SERIES_ORDER)
    if args.skip_t1_128:
        order = [a for a in order if a != "T1_128"]

    if args.from_arm:
        i = order.index(args.from_arm)
        order = order[i:]

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
        print(f"[chain_t] RUN {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(HERE))
        if rc != 0:
            print(f"[chain_t] FAIL {arm} exit={rc}", flush=True)
            sys.exit(rc)
    print("[chain_t] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
