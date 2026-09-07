# -*- coding: utf-8 -*-
"""带约束的 E1f 拟合（扩展 Exp34 e1f_core）。"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Callable  # keep if used

import numpy as np

_STEP = Path(__file__).resolve().parent
_A59 = _STEP.parent / "5070_challenge_mi_59ch_accpaper"
if str(_A59) not in sys.path:
    sys.path.insert(0, str(_A59))
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

from e1f_core import (  # noqa: E402
    E1fConfig,
    accuracy,
    apply_temperature,
    fuse_weighted,
    nll,
)
from scipy.optimize import minimize_scalar

from exp35_config import FusionConstraints  # noqa: E402


def fit_temperature_clamped(
    probs: np.ndarray,
    y: np.ndarray,
    *,
    t_min: float = 0.05,
    t_max: float = 10.0,
) -> float:
    def obj(t: float) -> float:
        return nll(apply_temperature(probs, t), y)

    res = minimize_scalar(obj, bounds=(float(t_min), float(t_max)), method="bounded")
    return float(res.x)


def _weight_grid(n: int, step: float = 0.1) -> list[tuple[float, ...]]:
    vals = np.arange(0.0, 1.0 + step * 0.5, step)
    grid: list[tuple[float, ...]] = []
    for ws in itertools.product(vals, repeat=n - 1):
        last = 1.0 - float(sum(ws))
        if last < -1e-9:
            continue
        w = tuple(float(x) for x in ws) + (float(last),)
        if abs(sum(w) - 1.0) > 1e-6 or any(x < -1e-9 for x in w):
            continue
        grid.append(w)
    return grid


def _rank_indices(member_accs: list[float]) -> tuple[int, int]:
    """返回 (best_idx, worst_idx)。"""
    order = sorted(range(len(member_accs)), key=lambda i: member_accs[i], reverse=True)
    return order[0], order[-1]


def _passes_constraints(
    w: tuple[float, ...],
    names: list[str],
    member_accs: list[float],
    cons: FusionConstraints,
) -> bool:
    wmap = {n: float(wi) for n, wi in zip(names, w)}
    if cons.w_eegnet_max is not None and "eegnet" in wmap:
        if wmap["eegnet"] > cons.w_eegnet_max + 1e-9:
            return False
    if cons.w_conformer_min is not None and "conformer" in wmap:
        if wmap["conformer"] + 1e-9 < cons.w_conformer_min:
            return False
    if cons.rank_prior or cons.rank_prior_relaxed:
        lo = 0.30 if cons.rank_prior_relaxed else 0.35
        hi = 0.20 if cons.rank_prior_relaxed else 0.15
        bi, wi = _rank_indices(member_accs)
        if w[bi] + 1e-9 < lo:
            return False
        if w[wi] > hi + 1e-9:
            return False
    return True


def fit_e1f_constrained(
    member_names: list[str],
    probs_list: list[np.ndarray],
    y: np.ndarray,
    cons: FusionConstraints | None = None,
) -> E1fConfig:
    """Val 上拟合 T + w；支持固定权重 / 温度夹紧 / 权重可行域。"""
    cons = cons or FusionConstraints(name="unconstrained")
    assert len(member_names) == len(probs_list)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    t_lo = cons.t_min if cons.t_min is not None else 0.05
    t_hi = cons.t_max if cons.t_max is not None else 10.0
    temps = [
        fit_temperature_clamped(p, y, t_min=t_lo, t_max=t_hi) for p in probs_list
    ]
    calibrated = [apply_temperature(p, t) for p, t in zip(probs_list, temps)]
    member_accs = [accuracy(c, y) for c in calibrated]

    if cons.fixed_weights is not None:
        if len(cons.fixed_weights) != len(member_names):
            raise ValueError("fixed_weights 长度与成员数不一致")
        w = tuple(float(x) for x in cons.fixed_weights)
        fused = fuse_weighted(calibrated, w)
        return E1fConfig(
            member_names=list(member_names),
            temperatures=temps,
            weights=list(w),
            smooth_radius=0,
            val_acc=float(accuracy(fused, y)),
        )

    # 单成员：权重恒 1
    if len(member_names) == 1:
        fused = calibrated[0]
        return E1fConfig(
            member_names=list(member_names),
            temperatures=temps,
            weights=[1.0],
            smooth_radius=0,
            val_acc=float(accuracy(fused, y)),
        )

    grid = _weight_grid(len(member_names))
    filtered = [
        w
        for w in grid
        if _passes_constraints(w, member_names, member_accs, cons)
    ]
    relaxed_note = False
    if not filtered and cons.rank_prior and not cons.rank_prior_relaxed:
        # 方案：放宽排名先验
        cons2 = FusionConstraints(
            w_eegnet_max=cons.w_eegnet_max,
            w_conformer_min=cons.w_conformer_min,
            rank_prior=True,
            rank_prior_relaxed=True,
            t_min=cons.t_min,
            t_max=cons.t_max,
            name=cons.name + "_relaxed",
        )
        filtered = [
            w
            for w in grid
            if _passes_constraints(w, member_names, member_accs, cons2)
        ]
        relaxed_note = bool(filtered)
        cons = cons2

    if not filtered:
        # 可行域空：回退无约束（登记）
        filtered = grid
        relaxed_note = True

    best_w = filtered[0]
    best_acc = -1.0
    for w in filtered:
        fused = fuse_weighted(calibrated, w)
        acc = accuracy(fused, y)
        if acc > best_acc:
            best_acc = acc
            best_w = w

    cfg = E1fConfig(
        member_names=list(member_names),
        temperatures=temps,
        weights=list(best_w),
        smooth_radius=0,
        val_acc=float(best_acc),
    )
    # 附加字段经 to_dict 不保留；调用方写 meta
    cfg._rankflip_relaxed = relaxed_note  # type: ignore[attr-defined]
    cfg._rankflip_cons_name = cons.name  # type: ignore[attr-defined]
    cfg._member_accs = member_accs  # type: ignore[attr-defined]
    return cfg


def fuse_with_config(probs_list: list[np.ndarray], cfg: E1fConfig) -> np.ndarray:
    calibrated = [
        apply_temperature(p, t) for p, t in zip(probs_list, cfg.temperatures)
    ]
    return fuse_weighted(calibrated, tuple(cfg.weights))
