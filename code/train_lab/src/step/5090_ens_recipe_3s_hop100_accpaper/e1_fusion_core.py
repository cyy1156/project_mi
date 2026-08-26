"""方案 26 · E1 推理侧满配：温度校准 / 加权 / 邻域平滑 / 置信早停。"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar

from s26_config import SMOOTH_R_CANDIDATES, WEIGHT_GRID_STEP


def probs_to_logits(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    return np.log(p)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-4)
    logits = probs_to_logits(probs) / t
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def nll_from_probs(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    idx = np.arange(len(y))
    return float(-np.log(p[idx, y.astype(int)]).mean())


def fit_temperature(probs: np.ndarray, y: np.ndarray, *, bounds=(0.05, 10.0)) -> float:
    def obj(t: float) -> float:
        return nll_from_probs(apply_temperature(probs, t), y)

    res = minimize_scalar(obj, bounds=bounds, method="bounded")
    return float(res.x)


def calibrate_members(
    members: list[dict[str, np.ndarray]],
    *,
    split: str = "val",
) -> list[float]:
    temps: list[float] = []
    for m in members:
        mask = m["split"] == split
        temps.append(fit_temperature(m["probs"][mask], m["y"][mask]))
    return temps


def apply_member_calibration(
    members: list[dict[str, np.ndarray]], temperatures: Iterable[float]
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for m, t in zip(members, temperatures):
        out.append(apply_temperature(m["probs"], t))
    return out


def _weight_grid(n_members: int, step: float = WEIGHT_GRID_STEP) -> list[tuple[float, ...]]:
    if n_members == 3:
        vals = np.arange(0.0, 1.0 + step * 0.5, step)
        grid: list[tuple[float, ...]] = []
        for w0 in vals:
            for w1 in vals:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                if abs(w0 + w1 + w2 - 1.0) > 1e-6:
                    continue
                if any(x < -1e-9 for x in (w0, w1, w2)):
                    continue
                grid.append((float(w0), float(w1), float(w2)))
        return grid
    step4 = max(step, 0.1)
    vals = np.arange(0.0, 1.0 + step4 * 0.5, step4)
    grid = []
    for ws in itertools.product(vals, repeat=n_members - 1):
        last = 1.0 - float(sum(ws))
        if last < -1e-9:
            continue
        w = tuple(float(x) for x in ws) + (last,)
        if abs(sum(w) - 1.0) > 1e-6:
            continue
        if any(x < -1e-9 for x in w):
            continue
        grid.append(w)
    return grid


def fuse_weighted(calibrated: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    stacked = np.stack(calibrated, axis=0)
    return np.tensordot(w, stacked, axes=(0, 0)).astype(np.float32)


def _trial_order(data: dict[str, np.ndarray]) -> dict[tuple[str, int], list[int]]:
    buckets: dict[tuple[str, int], list[int]] = {}
    for i in range(len(data["y"])):
        key = (str(data["subject"][i]), int(data["trial_id"][i]))
        buckets.setdefault(key, []).append(i)
    for key in buckets:
        buckets[key] = sorted(buckets[key], key=lambda j: float(data["t0_sec"][j]))
    return buckets


def temporal_smooth(probs: np.ndarray, data: dict[str, np.ndarray], radius: int) -> np.ndarray:
    if radius <= 0:
        return probs.copy()
    out = probs.copy()
    buckets = _trial_order(data)
    for idxs in buckets.values():
        n = len(idxs)
        for j, i in enumerate(idxs):
            lo = max(0, j - radius)
            hi = min(n, j + radius + 1)
            sl = idxs[lo:hi]
            out[i] = probs[sl].mean(axis=0)
    return out.astype(np.float32)


def pack_fused(base: dict[str, np.ndarray], probs: np.ndarray) -> dict[str, np.ndarray]:
    pred = probs.argmax(axis=1).astype(np.int64)
    pmax = probs.max(axis=1).astype(np.float32)
    out = dict(base)
    out["probs"] = probs
    out["pred"] = pred
    out["p_max"] = pmax
    return out


def align_members(members: list[dict[str, np.ndarray]]) -> None:
    base = members[0]
    n = len(base["y"])
    for m in members[1:]:
        assert len(m["y"]) == n
        assert np.array_equal(m["y"], base["y"])
        assert np.array_equal(m["trial_id"], base["trial_id"])
        assert np.array_equal(m["subject"], base["subject"])


def fuse_pipeline(
    members: list[dict[str, np.ndarray]],
    *,
    temperatures: list[float] | None,
    weights: tuple[float, ...],
    smooth_radius: int,
) -> dict[str, np.ndarray]:
    align_members(members)
    if temperatures is None:
        calibrated = [m["probs"] for m in members]
    else:
        calibrated = apply_member_calibration(members, temperatures)
    probs = fuse_weighted(calibrated, weights)
    probs = temporal_smooth(probs, members[0], smooth_radius)
    return pack_fused(members[0], probs)


@dataclass
class E1Config:
    temperatures: list[float]
    weights: tuple[float, ...]
    smooth_radius: int
    tau_conf: float | None = None
    consist_c: int | None = None

    def to_dict(self) -> dict:
        return {
            "temperatures": [float(t) for t in self.temperatures],
            "weights": [float(w) for w in self.weights],
            "smooth_radius": int(self.smooth_radius),
            "tau_conf": None if self.tau_conf is None else float(self.tau_conf),
            "consist_c": None if self.consist_c is None else int(self.consist_c),
        }


def _slice_members(members: list[dict[str, np.ndarray]], split: str) -> list[dict[str, np.ndarray]]:
    """网格搜索只在 val（或指定 split）子集上跑，避免每轮融合 30 万+ 窗。"""
    mask = members[0]["split"] == split
    out: list[dict[str, np.ndarray]] = []
    for m in members:
        sliced = {k: v[mask] for k, v in m.items()}
        out.append(sliced)
    return out


def acc_paper_on_rows(data: dict[str, np.ndarray]) -> float:
    from trial_metrics import aggregate_windows_to_trials

    trial = aggregate_windows_to_trials(
        data["y"],
        data["pred"],
        data["subject"],
        data["trial_id"],
        n_classes=3,
    )
    return float(trial["metrics"]["acc_paper"])


def acc_paper_for_split(data: dict[str, np.ndarray], split: str) -> float:
    from trial_metrics import aggregate_windows_to_trials

    m = data["split"] == split
    trial = aggregate_windows_to_trials(
        data["y"][m],
        data["pred"][m],
        data["subject"][m],
        data["trial_id"][m],
        n_classes=3,
    )
    return float(trial["metrics"]["acc_paper"])


def search_weights(
    members: list[dict[str, np.ndarray]],
    *,
    temperatures: list[float],
    smooth_radius: int,
    split: str = "val",
) -> tuple[float, ...]:
    best_w: tuple[float, ...] | None = None
    best_acc = -1.0
    val_members = _slice_members(members, split)
    for w in _weight_grid(len(members)):
        fused = fuse_pipeline(
            val_members,
            temperatures=temperatures,
            weights=w,
            smooth_radius=smooth_radius,
        )
        acc = acc_paper_on_rows(fused)
        if acc > best_acc:
            best_acc = acc
            best_w = w
    assert best_w is not None
    return best_w


def search_smooth_radius(
    members: list[dict[str, np.ndarray]],
    *,
    temperatures: list[float],
    weights: tuple[float, ...],
    split: str = "val",
    candidates: Iterable[int] = SMOOTH_R_CANDIDATES,
) -> int:
    best_r = 0
    best_acc = -1.0
    val_members = _slice_members(members, split)
    for r in candidates:
        fused = fuse_pipeline(
            val_members,
            temperatures=temperatures,
            weights=weights,
            smooth_radius=int(r),
        )
        acc = acc_paper_on_rows(fused)
        if acc > best_acc:
            best_acc = acc
            best_r = int(r)
    return best_r


def fit_e1_config(
    members: list[dict[str, np.ndarray]],
    *,
    use_temp: bool = True,
    use_weights: bool = True,
    use_smooth: bool = True,
    n_members: int | None = None,
) -> E1Config:
    n = n_members or len(members)
    uniform = tuple([1.0 / n] * n)
    temps = calibrate_members(members) if use_temp else [1.0] * len(members)
    weights = search_weights(members, temperatures=temps, smooth_radius=0) if use_weights else uniform
    radius = search_smooth_radius(members, temperatures=temps, weights=weights) if use_smooth else 0
    return E1Config(temperatures=temps, weights=weights, smooth_radius=radius)


def simulate_conf_early_stop(
    data: dict[str, np.ndarray],
    *,
    tau_conf: float,
) -> dict[str, np.ndarray]:
    """全 split 试次内按 t0 扫描，p_max >= tau 即提交。"""
    out = dict(data)
    pred = data["pred"].copy()
    buckets = _trial_order(data)
    for gidxs in buckets.values():
        picked = gidxs[-1]
        for j in gidxs:
            if float(data["p_max"][j]) >= tau_conf:
                picked = j
                break
        label = int(data["pred"][picked])
        for j in gidxs:
            pred[j] = label
    out["pred"] = pred
    return out


def search_tau_conf(data: dict[str, np.ndarray], *, split: str = "val") -> float:
    from s26_config import TAU_CONF_GRID

    best_tau = TAU_CONF_GRID[0]
    best_acc = -1.0
    mask = data["split"] == split
    val_data = {k: v[mask] for k, v in data.items()}
    for tau in TAU_CONF_GRID:
        sub = simulate_conf_early_stop(val_data, tau_conf=float(tau))
        acc = acc_paper_on_rows(sub)
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)
    return best_tau
