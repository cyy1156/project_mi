"""方案 26 · E2b Riemannian 切空间 + L2 逻辑回归成员。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from e2_common import run_riemann_e2b  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme26 E2b · Riemannian tangent member")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--band-split", action="store_true", help="μ/β 分带协方差 → 72 维")
    args = p.parse_args()
    run_riemann_e2b(max_folds=args.max_folds, band_split=args.band_split)


if __name__ == "__main__":
    main()
