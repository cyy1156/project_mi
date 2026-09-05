"""ShallowFBCSPNet + 前置 SE / ECA。"""

from __future__ import annotations

import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet

from attention_front import ECA1d, SE1d


class ShallowFrontAttn(nn.Module):
    """attn(X) → ShallowFBCSPNet；不改骨干超参。"""

    def __init__(self, attn: nn.Module, backbone: ShallowFBCSPNet):
        super().__init__()
        self.attn = attn
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.attn(x))


def _make_shallow(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float,
) -> ShallowFBCSPNet:
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
        final_conv_length="auto",
    )


def build_s0(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float,
) -> nn.Module:
    return _make_shallow(n_chans, n_times, n_outputs, drop_prob)


def build_shallow_se(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float,
    *,
    reduction: int = 2,
) -> nn.Module:
    return ShallowFrontAttn(
        SE1d(n_chans, reduction=reduction),
        _make_shallow(n_chans, n_times, n_outputs, drop_prob),
    )


def build_shallow_eca(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float,
    *,
    k_size: int = 3,
) -> nn.Module:
    return ShallowFrontAttn(
        ECA1d(n_chans, k_size=k_size),
        _make_shallow(n_chans, n_times, n_outputs, drop_prob),
    )


def count_attn_params(model: nn.Module) -> int:
    if isinstance(model, ShallowFrontAttn):
        return sum(p.numel() for p in model.attn.parameters())
    return 0
