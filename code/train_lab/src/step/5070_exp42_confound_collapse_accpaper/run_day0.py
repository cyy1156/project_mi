"""Exp42 Day0 ???P0 ? ?? ? Leave-Next ?? ? A/B/C/D/E ? ????

???????????
  cd D:\\MI\\code\\train_lab\\src\\step\\5070_exp42_confound_collapse_accpaper
  python run_day0.py --workers 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    from cohort_map import build_cohort_map, main as p0_main
    from extract_features import run_extract
    from parse_leave_next import run_parse
    from arms_day0 import run_arms
    from summary_42 import write_registry

    cohort_path = p0_main()
    cohort = build_cohort_map()
    run_parse(cohort)
    if not args.skip_extract:
        run_extract(cohort, max_workers=args.workers)
    run_arms()
    write_registry()
    print("[day0] DONE", cohort_path)


if __name__ == "__main__":
    main()
