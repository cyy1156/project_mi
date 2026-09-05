# -*- coding: utf-8 -*-
"""Thin wrapper: prefer arm_A_jackknife.py for real runs."""
from __future__ import annotations

import argparse

from arm_A_jackknife import DEFAULT_ARMS, DEFAULT_SEEDS, run_arm_A


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp42 Arm A (delegates to arm_A_jackknife)")
    ap.add_argument("--person", type=str, default="", help="alias of --people single id")
    ap.add_argument("--people", type=str, default="")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS))
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=0)
    args = ap.parse_args()
    people_s = args.people or args.person
    people = [x.strip() for x in people_s.split(",") if x.strip()] or None
    run_arm_A(
        people=people,
        arms=tuple(x.strip() for x in args.arms.split(",") if x.strip()),
        seeds=tuple(int(x) for x in args.seeds.split(",") if x.strip()),
        device=args.device or None,
        dry_run=args.dry_run,
        max_jobs=args.max_jobs or None,
    )


if __name__ == "__main__":
    main()
