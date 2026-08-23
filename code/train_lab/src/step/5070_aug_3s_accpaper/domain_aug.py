"""域增广（z-score 之后施加，仅训练期）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import torch

# G1 主跑：#1+#2+#3
AUG_MAIN = ("noise", "time_shift", "ch_dropout")
AUG_REPORT = ("ch_noise_replace", "global_scale", "mixup")
AUG_FT_LIGHT = ("noise", "time_shift")  # G3：σ/时移减半


@dataclass(frozen=True)
class AugConfig:
    enabled: bool = False
    p_apply: float = 0.5
    noise_sigmas: tuple[float, ...] = (0.1, 0.2, 0.3)
    time_shift_max: int = 25
    ch_dropout_p: float = 0.1
    ch_noise_replace_p: float = 0.1
    global_scale_range: tuple[float, float] = (0.8, 1.25)
    mixup_alpha: float = 0.2
    ops: tuple[str, ...] = AUG_MAIN
    # G3 轻增广：在 parse 时设 light=True
    light: bool = False

    def effective(self) -> AugConfig:
        if not self.light:
            return self
        return replace(
            self,
            p_apply=0.3,
            noise_sigmas=tuple(s * 0.5 for s in self.noise_sigmas),
            time_shift_max=max(1, self.time_shift_max // 2),
        )


def aug_config_g1() -> AugConfig:
    return AugConfig(enabled=True, ops=AUG_MAIN)


def aug_config_g3_ft() -> AugConfig:
    return AugConfig(enabled=True, ops=AUG_FT_LIGHT, light=True)


def aug_config_from_spec(spec: str) -> AugConfig:
    s = (spec or "off").strip().lower()
    if s in ("off", "none", "0", "false"):
        return AugConfig(enabled=False)
    if s in ("g1", "main"):
        return aug_config_g1()
    if s in ("g3", "ft_light", "light"):
        return aug_config_g3_ft()
    if s == "report":
        return AugConfig(
            enabled=True,
            ops=tuple(AUG_MAIN) + tuple(AUG_REPORT),
        )
    ops = tuple(x.strip() for x in s.split(",") if x.strip())
    return AugConfig(enabled=True, ops=ops)


def _rng_from_item(seed_base: int, index: int) -> np.random.Generator:
    return np.random.default_rng(int(seed_base) + int(index) * 9973)


def apply_domain_aug_np(
    x: np.ndarray,
    cfg: AugConfig,
    *,
    seed: int,
    index: int,
    y: int | None = None,
    y_pool: np.ndarray | None = None,
    x_pool: np.ndarray | None = None,
) -> np.ndarray:
    """x: (8, T) float32，已 z-score。"""
    cfg = cfg.effective()
    if not cfg.enabled:
        return x
    rng = _rng_from_item(seed, index)
    if rng.random() > cfg.p_apply:
        return x
    out = np.array(x, dtype=np.float32, copy=True)
    # 每窗随机选一种增广（方案冻结集内）
    op = str(rng.choice(list(cfg.ops)))
    if op == "noise":
        sigma = float(rng.choice(cfg.noise_sigmas))
        out += rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
    elif op == "time_shift":
        shift = int(rng.integers(-cfg.time_shift_max, cfg.time_shift_max + 1))
        if shift != 0:
            out = np.roll(out, shift, axis=-1)
    elif op == "ch_dropout":
        if rng.random() < cfg.ch_dropout_p:
            ch = int(rng.integers(0, out.shape[0]))
            out[ch, :] = 0.0
    elif op == "ch_noise_replace":
        if rng.random() < cfg.ch_noise_replace_p:
            ch = int(rng.integers(0, out.shape[0]))
            out[ch, :] = rng.normal(0.0, 1.0, size=out.shape[1]).astype(np.float32)
    elif op == "global_scale":
        scale = float(rng.uniform(*cfg.global_scale_range))
        out *= scale
    elif op == "mixup":
        if y_pool is not None and x_pool is not None and len(x_pool) > 1:
            lam = float(rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
            j = int(rng.integers(0, len(x_pool)))
            if int(y_pool[j]) == int(y) if y is not None else True:
                out = (lam * out + (1.0 - lam) * x_pool[j]).astype(np.float32)
    else:
        raise ValueError(f"未知增广 op={op!r}")
    return out


def apply_domain_aug_torch(
    x: torch.Tensor,
    cfg: AugConfig,
    *,
    seed: int,
    index: int,
) -> torch.Tensor:
    arr = x.detach().cpu().numpy()
    out = apply_domain_aug_np(arr, cfg, seed=seed, index=index)
    return torch.from_numpy(out)


def aug_config_to_meta(cfg: AugConfig) -> dict[str, Any]:
    d = asdict(cfg.effective())
    d["ops"] = list(d["ops"])
    return d
