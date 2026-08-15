"""方案 09 · S3 读出头：在默认 Shallow 骨干上替换 final_layer。"""

from __future__ import annotations

import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet


class _TemporalStatsHead(nn.Module):
    """对 (B,C,T,1) 做 mean/std/max 拼接 → Dropout → Linear。"""

    def __init__(self, n_filters: int, n_outputs: int, drop_prob: float):
        super().__init__()
        self.drop = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_filters * 3, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, 1) after SafeLog+Dropout in shallow trunk
        if x.ndim == 4:
            x = x.squeeze(-1)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        mx = x.amax(dim=-1)
        h = torch.cat([mean, std, mx], dim=1)
        return self.fc(self.drop(h))


def _make_base(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float,
    *,
    filter_time_length: int = 25,
    n_filters_time: int = 40,
    n_filters_spat: int = 40,
    pool_time_length: int = 75,
    pool_time_stride: int = 15,
) -> ShallowFBCSPNet:
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
        n_filters_time=n_filters_time,
        filter_time_length=filter_time_length,
        n_filters_spat=n_filters_spat,
        pool_time_length=pool_time_length,
        pool_time_stride=pool_time_stride,
        final_conv_length="auto",
    )


def build_s3_mlp(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float,
    hidden: int = 64,
) -> nn.Module:
    m = _make_base(n_chans, n_times, n_outputs, drop_prob)
    n_f = int(m.final_layer[0].in_channels) if hasattr(m.final_layer[0], "in_channels") else 40
    m.final_layer = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(drop_prob),
        nn.Linear(n_f, hidden),
        nn.GELU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden, n_outputs),
    )
    return m


def build_s3_stats(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float,
) -> nn.Module:
    m = _make_base(n_chans, n_times, n_outputs, drop_prob)
    n_f = int(m.final_layer[0].in_channels) if hasattr(m.final_layer[0], "in_channels") else 40
    m.final_layer = _TemporalStatsHead(n_f, n_outputs, drop_prob)
    return m
