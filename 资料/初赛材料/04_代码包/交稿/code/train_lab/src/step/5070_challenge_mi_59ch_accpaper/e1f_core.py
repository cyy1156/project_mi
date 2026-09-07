"""E1f 核心：温度校准 + 权重网格（单窗协议下 Acc≡Acc_paper；smooth 默认关）。"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def apply_temperature(probs: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    t = max(float(temperature), 1e-4)
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    logits = np.log(p) / t
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return float(-np.log(p[np.arange(len(y)), y.astype(int)]).mean())


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    def obj(t: float) -> float:
        return nll(apply_temperature(probs, t), y)

    res = minimize_scalar(obj, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def fuse_weighted(calibrated: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    return np.tensordot(w, np.stack(calibrated, axis=0), axes=(0, 0)).astype(np.float32)


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


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == y.astype(int)).mean()) if len(y) else 0.0


@dataclass
class E1fConfig:
    member_names: list[str]
    temperatures: list[float]
    weights: list[float]
    smooth_radius: int = 0
    val_acc: float = 0.0

    def to_dict(self) -> dict:
        return {
            "member_names": list(self.member_names),
            "temperatures": [float(t) for t in self.temperatures],
            "weights": [float(w) for w in self.weights],
            "smooth_radius": int(self.smooth_radius),
            "val_acc": float(self.val_acc),
        }


def fit_e1f(
    member_names: list[str],
    probs_list: list[np.ndarray],
    y: np.ndarray,
) -> E1fConfig:
    """在 Val 上拟合 T + 权重（smooth_radius=0：官方单窗无需邻域平滑）。"""
    assert len(member_names) == len(probs_list)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    temps = [fit_temperature(p, y) for p in probs_list]
    calibrated = [apply_temperature(p, t) for p, t in zip(probs_list, temps)]
    best_w = tuple([1.0 / len(member_names)] * len(member_names))
    best_acc = -1.0
    for w in _weight_grid(len(member_names)):
        fused = fuse_weighted(calibrated, w)
        acc = accuracy(fused, y)
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return E1fConfig(
        member_names=list(member_names),
        temperatures=temps,
        weights=list(best_w),
        smooth_radius=0,
        val_acc=float(best_acc),
    )


def fuse_with_config(probs_list: list[np.ndarray], cfg: E1fConfig) -> np.ndarray:
    calibrated = [
        apply_temperature(p, t) for p, t in zip(probs_list, cfg.temperatures)
    ]
    return fuse_weighted(calibrated, tuple(cfg.weights))
