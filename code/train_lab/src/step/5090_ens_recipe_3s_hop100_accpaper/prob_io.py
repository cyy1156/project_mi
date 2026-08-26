"""方案 26 · 加载成员 prob dump 与路径工具。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG24 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
for p in (HERE, PKG24):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from prob_dump import load_prob_dump, merge_prob_dumps  # noqa: E402


def load_run_three(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    dumps = sorted(run_dir.glob("fold*/prob_dump_three.csv"))
    if not dumps:
        dumps = sorted(run_dir.glob("fold*/prob_dump_three_*.csv"))
    if not dumps:
        raise FileNotFoundError(f"no prob dumps under {run_dir}")
    return merge_prob_dumps(dumps)


def load_members(run_dirs: list[Path]) -> list[dict]:
    return [load_run_three(p) for p in run_dirs]
