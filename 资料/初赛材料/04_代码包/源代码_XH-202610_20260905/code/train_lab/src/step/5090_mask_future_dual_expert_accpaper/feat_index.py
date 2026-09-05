"""§3.2.1 特征时间索引硬规格。"""
from __future__ import annotations

import torch


def feat_index(t: int, n_times: int = 1000, t_prime: int = 61) -> int:
    return int(round(t / (n_times - 1) * (t_prime - 1)))


def segment_indices(n_times: int = 1000, t_prime: int = 61) -> tuple[list[int], list[int]]:
    # 禁止: cut = int(0.6 * t_prime)
    i_split = feat_index(600, n_times, t_prime)  # 36
    i_vis = list(range(0, i_split))  # 0..35
    i_future = list(range(i_split, t_prime))  # 36..60
    return i_vis, i_future


def assert_default_map() -> None:
    assert feat_index(600) == 36
    i_vis, i_fut = segment_indices()
    assert i_vis[-1] == 35 and i_fut[0] == 36
    assert len(i_vis) + len(i_fut) == 61


@torch.no_grad()
def assert_future_perturbation(
    encoder: torch.nn.Module,
    *,
    i_vis: list[int],
    i_fut: list[int],
    n_chans: int = 8,
    n_times: int = 1000,
    future_pts: int | None = None,
    ratio_min: float = 3.0,
    noise_std: float = 1.0,
    seed: int = 0,
) -> float:
    """
    扰动 future 段后，‖ΔZ_fut‖ / ‖ΔZ_vis‖ ≥ ratio_min（协议停训门槛）。
    返回实测比值。
    """
    if not i_fut:
        raise AssertionError("i_fut 为空，无法做 future 扰动检验")
    if future_pts is None:
        future_pts = 200 if n_times == 800 else 400
    g = torch.Generator().manual_seed(int(seed))
    x0 = torch.randn(2, n_chans, n_times, generator=g)
    x1 = x0.clone()
    x1[..., -future_pts:] = x1[..., -future_pts:] + noise_std * torch.randn(
        2, n_chans, future_pts, generator=g
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
