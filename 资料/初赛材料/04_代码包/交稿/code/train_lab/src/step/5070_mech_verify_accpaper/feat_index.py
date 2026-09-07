"""§3.2.1 特征时间索引 · 支持可变 n_times / 可见段长度。"""
from __future__ import annotations

import torch

from geometry import GeomSpec, get_geom


def feat_index(t: int, n_times: int = 1000, t_prime: int = 61) -> int:
    if n_times <= 1:
        return 0
    return int(round(t / (n_times - 1) * (t_prime - 1)))


def segment_indices(
    n_times: int = 1000,
    *,
    vis_pts: int | None = None,
    t_prime: int = 61,
) -> tuple[list[int], list[int]]:
    if vis_pts is None:
        vis_pts = 600 if n_times >= 1000 else n_times
    vis_pts = min(int(vis_pts), int(n_times))
    i_split = feat_index(vis_pts, n_times, t_prime)
    i_split = max(1, min(i_split, t_prime - 1))
    i_vis = list(range(0, i_split))
    i_future = list(range(i_split, t_prime))
    return i_vis, i_future


def segment_indices_for_geom(geom_id: str, t_prime: int = 61) -> tuple[list[int], list[int]]:
    g = get_geom(geom_id)
    return segment_indices(g.total_pts, vis_pts=g.vis_pts, t_prime=t_prime)


def assert_default_map() -> None:
    assert feat_index(600, 1000) == 36
    i_vis, i_fut = segment_indices(1000, vis_pts=600)
    assert i_vis[-1] == 35 and i_fut[0] == 36
    assert len(i_vis) + len(i_fut) == 61
    i600_vis, i600_fut = segment_indices(600, vis_pts=600, t_prime=61)
    assert len(i600_vis) + len(i600_fut) == 61


@torch.no_grad()
def assert_future_perturbation(
    encoder: torch.nn.Module,
    *,
    i_vis: list[int],
    i_fut: list[int],
    n_chans: int = 8,
    n_times: int = 1000,
    future_pts: int = 400,
    ratio_min: float = 3.0,
    noise_std: float = 1.0,
    seed: int = 0,
) -> float:
    if not i_fut:
        return float("nan")
    g = torch.Generator().manual_seed(int(seed))
    x0 = torch.randn(2, n_chans, n_times, generator=g)
    x1 = x0.clone()
    fp = min(int(future_pts), int(n_times))
    x1[..., -fp:] = x1[..., -fp:] + noise_std * torch.randn(
        2, n_chans, fp, generator=g
    )

    def _pool(feat: torch.Tensor, idx: list[int]) -> torch.Tensor:
        if feat.ndim == 4:
            feat = feat.squeeze(-1)
        return feat[:, :, idx].mean(dim=-1)

    was_training = encoder.training
    encoder.eval()
    f0 = encoder.forward_features(x0)
    f1 = encoder.forward_features(x1)
    if was_training:
        encoder.train()

    dv = (_pool(f1, i_vis) - _pool(f0, i_vis)).norm().item()
    df = (_pool(f1, i_fut) - _pool(f0, i_fut)).norm().item()
    ratio = df / max(dv, 1e-8)
    assert ratio >= ratio_min, (
        f"§3.2.1 future 扰动失败: ‖ΔZ_fut‖/‖ΔZ_vis‖={ratio:.3f} < {ratio_min}"
    )
    return float(ratio)
