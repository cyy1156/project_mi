"""方案 23 · pf1000 运行时几何切片（G2s / G15 / G1s / G600）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GeomSpec:
    geom_id: str
    past_pts: int
    cur_pts: int
    future_pts: int
    total_pts: int

    @property
    def vis_pts(self) -> int:
        return self.past_pts + self.cur_pts


GEOMETRIES: dict[str, GeomSpec] = {
    "G2s": GeomSpec("G2s", 100, 500, 400, 1000),
    "G15": GeomSpec("G15", 100, 375, 400, 875),
    "G1s": GeomSpec("G1s", 100, 250, 400, 750),
    "G600": GeomSpec("G600", 100, 500, 0, 600),
}


def get_geom(geom_id: str) -> GeomSpec:
    g = GEOMETRIES.get(geom_id)
    if g is None:
        raise KeyError(f"unknown geometry {geom_id!r}; expected one of {list(GEOMETRIES)}")
    return g


def slice_pf1000(x: np.ndarray | torch.Tensor, geom_id: str) -> np.ndarray | torch.Tensor:
    """从 pf1000 1000pt 画布裁切几何；输入末维须为 1000。"""
    g = get_geom(geom_id)
    if g.geom_id == "G2s":
        return x[..., :1000]
    start_cur = g.past_pts
    end = start_cur + g.cur_pts + g.future_pts
    return x[..., :end]


def make_masked(x_full: torch.Tensor, geom_id: str) -> torch.Tensor:
    """future 段置零（在线契约）。"""
    g = get_geom(geom_id)
    xm = x_full.clone()
    if g.future_pts > 0:
        xm[..., -g.future_pts :] = 0.0
    return xm


def resolve_input(
    x_full: torch.Tensor,
    x_mask: torch.Tensor,
    *,
    geom_id: str,
    oracle: bool,
) -> torch.Tensor:
    if oracle:
        return x_full
    return x_mask


def future_perturb_tail_pts(geom_id: str) -> int:
    g = get_geom(geom_id)
    return g.future_pts
