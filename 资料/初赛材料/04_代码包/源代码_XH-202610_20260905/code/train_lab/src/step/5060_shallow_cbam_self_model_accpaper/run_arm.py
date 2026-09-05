"""Scheme 15 arms: S0 / A1 / A2 / B1 / B2 on self_model Shallow+CBAM.

Usage:
  python run_arm.py --arm A1                 # fold0 (default for A1/A2)
  python run_arm.py --arm A1 --max-folds 0   # full 5-fold
  python run_arm.py --arm A2
  python run_arm.py --arm S0 --max-folds 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SELF_MODEL = REPO / "self_model"
sys.path.insert(0, str(SELF_MODEL))

from shallowfbcsp import ATTN_BY_ARM, build_model  # noqa: E402
from task_runner import run_baseline_main  # noqa: E402

# default max_folds: 1 for probe arms, 0 (all) if user passes --max-folds
ARMS = {
    "S0": dict(note="S0 self_model Shallow · no attn"),
    "A1": dict(note="A1 ConvTime→full CBAM→ConvSpat (electrode-overlap control)"),
    "A2": dict(note="A2 split: channel@Time + temporal-spatial@BN (primary)"),
    "B1": dict(note="B1 appendix CBAM after BN"),
    "B2": dict(note="B2 appendix CBAM after Drop"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="15 shallow CBAM self_model arm")
    p.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    args, rest = p.parse_known_args()

    # A1/A2/B* default fold0 unless user sets --max-folds
    if args.arm != "S0" and "--max-folds" not in rest:
        rest = [*rest, "--max-folds", "1"]
    sys.argv = [sys.argv[0], *rest]

    arm = args.arm
    attn = ATTN_BY_ARM[arm]

    def build(n_chans, n_times, n_outputs, drop_prob) -> nn.Module:
        return build_model(n_chans, n_times, n_outputs, drop_prob, arm=arm)

    run_baseline_main(
        model_name=f"shallow_cbam_{arm.lower()}",
        build_model=build,
        input_kind="time",
        structure_note=ARMS[arm]["note"] + f" | attn={attn}",
        extra_meta={
            "scheme": "15",
            "arm": arm,
            "attn": attn,
            "backbone": "self_model.shallowfbcsp",
            "accpaper": True,
            "bypass": True,
        },
    )


if __name__ == "__main__":
    main()
