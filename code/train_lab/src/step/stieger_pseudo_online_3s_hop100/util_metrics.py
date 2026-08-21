"""公用：JSON / mean±std / 导入游戏门控与段级指标。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
GAME = STEP / "game_pseudo_online_hop100"
for p in (str(GAME), str(STEP), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(STEP))
sys.path.insert(0, str(GAME))
sys.path.insert(0, str(HERE))

from gated_segment_metrics import aggregate_windows_to_segments_gated  # noqa: E402
from eval_metrics import aggregate_windows_to_segments  # noqa: E402
from online_gate import build_gate_keeps, gate_stats  # noqa: E402


def mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


__all__ = [
    "aggregate_windows_to_segments",
    "aggregate_windows_to_segments_gated",
    "build_gate_keeps",
    "gate_stats",
    "jsonable",
    "mean_std",
]
