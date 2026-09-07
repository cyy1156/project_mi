"""加载 teachable_v1 清单 / 窗 mask。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DEFAULT_JSON = REPO / "find_best_trail" / "out" / "teachable_trials_v1.json"
DEFAULT_MASK = REPO / "find_best_trail" / "out" / "teachable_window_masks_v1.npz"


def resolve_teachable_paths(
    json_path: str | Path | None = None,
    mask_path: str | Path | None = None,
) -> tuple[Path, Path]:
    jp = Path(json_path) if json_path else DEFAULT_JSON
    mp = Path(mask_path) if mask_path else DEFAULT_MASK
    return jp, mp


def load_masks(
    n_windows: int,
    mask_path: Path | None = None,
) -> dict[str, np.ndarray]:
    mp = mask_path or DEFAULT_MASK
    if not mp.is_file():
        raise FileNotFoundError(
            f"缺少窗 mask：{mp}\n请先跑 B0：python -m src.datasets.openbmi.export_teachable_trials"
        )
    z = np.load(mp, allow_pickle=True)
    out = {}
    for k in ("teachable", "template_grade", "obvious12", "high_lat_eval"):
        arr = np.asarray(z[k], dtype=bool).reshape(-1)
        if len(arr) != n_windows:
            raise ValueError(f"{mp} {k} len={len(arr)} != n_windows={n_windows}")
        out[k] = arr
    return out


def load_trial_table(json_path: Path | None = None) -> dict:
    jp = json_path or DEFAULT_JSON
    if not jp.is_file():
        raise FileNotFoundError(f"缺少清单：{jp}")
    return json.loads(jp.read_text(encoding="utf-8"))


def subset_mask(name: str, masks: dict[str, np.ndarray]) -> np.ndarray:
    """R0=None 调用方自行处理；R1/R2/R3 对应 obvious12 / high_lat_eval / teachable。"""
    key = {
        "R1": "obvious12",
        "R2": "high_lat_eval",
        "R3": "teachable",
        "obvious12": "obvious12",
        "high_lat_eval": "high_lat_eval",
        "teachable": "teachable",
    }[name]
    return masks[key]
