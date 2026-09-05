"""方案 26 · E2a KAN 带功率成员五折。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from e2_common import run_kan_e2a  # noqa: E402
from fast_kan import FastKANClassifier  # noqa: E402


def build_model(in_dim: int, n_outputs: int, drop_prob: float):
    _ = drop_prob
    return FastKANClassifier(in_dim, n_outputs, hidden=64, grid=5)


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme26 E2a · KAN bandpower member")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=0)
    args = p.parse_args()
    run_kan_e2a(
        build_model=build_model,
        max_folds=args.max_folds,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
