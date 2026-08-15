"""Standalone ShallowFBCSPNet: modules first, then assemble. No braindecode.

attn modes (scheme 15):
  None        — S0 baseline
  cbam_time   — A1: full CBAM after ConvTime (may overlap ConvSpat on electrodes)
  split       — A2: channel after ConvTime + temporal-spatial after BN
  cbam_mid    — B1 appendix: CBAM after BN (time as spatial axis)
  cbam_late   — B2 appendix: CBAM after Drop
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Ensure4d(nn.Module):
    """(B,C,T) -> (B,C,T,1)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.ndim < 4:
            x = x.unsqueeze(-1)
        return x


class DimShuffle(nn.Module):
    """(B,C,T,1) -> (B,1,T,C)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 2, 1).contiguous()


class ConvTime(nn.Module):
    """Temporal conv: (B,1,T,C) -> (B,F_t,T',C)."""

    def __init__(self, n_filters_time: int = 40, filter_time_length: int = 25) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            1, n_filters_time, (filter_time_length, 1), stride=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvSpat(nn.Module):
    """Spatial conv: (B,F_t,T',C) -> (B,F_s,T',1)."""

    def __init__(
        self,
        n_chans: int,
        n_filters_time: int = 40,
        n_filters_spat: int = 40,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            n_filters_time, n_filters_spat, (1, n_chans), stride=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class SafeLog(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class SqueezeFinalOutput(nn.Module):
    """(B,C,T,1) -> (B,C) or (B,C,T)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., 0]
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x


class FinalClassifier(nn.Module):
    def __init__(self, n_filters: int, n_outputs: int, final_conv_length: int) -> None:
        super().__init__()
        self.conv_classifier = nn.Conv2d(
            n_filters, n_outputs, (final_conv_length, 1), bias=True
        )
        self.squeeze = SqueezeFinalOutput()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.squeeze(self.conv_classifier(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(int(channels) // max(int(reduction), 1), 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * (att * 2.0)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * (att * 2.0)


class CBAM(nn.Module):
    """Channel → Spatial (paper order)."""

    def __init__(
        self, channels: int, reduction: int = 16, spatial_kernel: int = 7
    ) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


def cbam_on_time_axis(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """(B,C,T,1) <-> (B,C,1,T) so spatial attn acts on time."""
    y = x.permute(0, 1, 3, 2).contiguous()
    y = module(y)
    return y.permute(0, 1, 3, 2).contiguous()


class ShallowFBCSPNet(nn.Module):
    """Schirrmeister 2017; defaults aligned with braindecode."""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        final_conv_length: int | str = "auto",
        pool_mode: str = "mean",
        batch_norm: bool = True,
        batch_norm_alpha: float = 0.1,
        drop_prob: float = 0.5,
        attn: str | None = None,
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_outputs = int(n_outputs)
        self.n_times = int(n_times)
        self.attn = None if attn in (None, "", "none", "S0") else str(attn)

        pool_cls = {"max": nn.MaxPool2d, "mean": nn.AvgPool2d}[pool_mode]

        self.ensuredims = Ensure4d()
        self.dimshuffle = DimShuffle()
        self.conv_time = ConvTime(n_filters_time, filter_time_length)
        self.conv_spat = ConvSpat(
            n_chans=self.n_chans,
            n_filters_time=n_filters_time,
            n_filters_spat=n_filters_spat,
            bias=not batch_norm,
        )
        self.bnorm = (
            nn.BatchNorm2d(n_filters_spat, momentum=batch_norm_alpha, affine=True)
            if batch_norm
            else nn.Identity()
        )
        self.conv_nonlin_exp = Square()
        self.pool = pool_cls(
            kernel_size=(pool_time_length, 1),
            stride=(pool_time_stride, 1),
        )
        self.pool_nonlin_exp = SafeLog()
        self.drop = nn.Dropout(p=drop_prob)

        self.cbam_time = None
        self.channel_after_time = None
        self.temporal_spatial = None
        self.cbam_mid = None
        self.cbam_late = None
        r, k = int(cbam_reduction), int(cbam_spatial_kernel)
        if self.attn == "cbam_time":
            self.cbam_time = CBAM(n_filters_time, reduction=r, spatial_kernel=k)
        elif self.attn == "split":
            self.channel_after_time = ChannelAttention(n_filters_time, reduction=r)
            self.temporal_spatial = SpatialAttention(kernel_size=k)
        elif self.attn == "cbam_mid":
            self.cbam_mid = CBAM(n_filters_spat, reduction=r, spatial_kernel=k)
        elif self.attn == "cbam_late":
            self.cbam_late = CBAM(n_filters_spat, reduction=r, spatial_kernel=k)
        elif self.attn is not None:
            raise ValueError(
                f"unknown attn={self.attn!r}; "
                "use None|cbam_time|split|cbam_mid|cbam_late"
            )

        if final_conv_length == "auto":
            self.eval()
            with torch.inference_mode():
                feat = self.forward_features(torch.zeros(1, self.n_chans, self.n_times))
                final_conv_length = int(feat.shape[2])
            self.train()
        self.final_conv_length = int(final_conv_length)

        self.final_layer = FinalClassifier(
            n_filters=n_filters_spat,
            n_outputs=self.n_outputs,
            final_conv_length=self.final_conv_length,
        )
        self._init_weights(batch_norm=batch_norm)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensuredims(x)
        x = self.dimshuffle(x)
        x = self.conv_time(x)
        if self.cbam_time is not None:
            x = self.cbam_time(x)
        if self.channel_after_time is not None:
            x = self.channel_after_time(x)
        x = self.conv_spat(x)
        x = self.bnorm(x)
        if self.temporal_spatial is not None:
            x = cbam_on_time_axis(self.temporal_spatial, x)
        if self.cbam_mid is not None:
            x = cbam_on_time_axis(self.cbam_mid, x)
        x = self.conv_nonlin_exp(x)
        x = self.pool(x)
        x = self.pool_nonlin_exp(x)
        x = self.drop(x)
        if self.cbam_late is not None:
            x = cbam_on_time_axis(self.cbam_late, x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.forward_features(x))

    def _init_weights(self, *, batch_norm: bool) -> None:
        init.xavier_uniform_(self.conv_time.conv.weight, gain=1)
        init.constant_(self.conv_time.conv.bias, 0)
        init.xavier_uniform_(self.conv_spat.conv.weight, gain=1)
        if not batch_norm:
            init.constant_(self.conv_spat.conv.bias, 0)
        if batch_norm and isinstance(self.bnorm, nn.BatchNorm2d):
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.final_layer.conv_classifier.weight, gain=1)
        init.constant_(self.final_layer.conv_classifier.bias, 0)


ATTN_BY_ARM = {
    "S0": None,
    "A1": "cbam_time",
    "A2": "split",
    "B1": "cbam_mid",
    "B2": "cbam_late",
}


def build_model(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float,
    *,
    attn: str | None = None,
    arm: str | None = None,
    cbam_reduction: int = 16,
    cbam_spatial_kernel: int = 7,
) -> nn.Module:
    if arm is not None:
        if arm not in ATTN_BY_ARM:
            raise ValueError(f"unknown arm={arm!r}; choose {list(ATTN_BY_ARM)}")
        attn = ATTN_BY_ARM[arm]
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
        attn=attn,
        cbam_reduction=cbam_reduction,
        cbam_spatial_kernel=cbam_spatial_kernel,
    )


if __name__ == "__main__":
    for arm in ("S0", "A1", "A2", "B1", "B2"):
        m = build_model(8, 500, 2, 0.5, arm=arm)
        y = m(torch.randn(4, 8, 500))
        n = sum(p.numel() for p in m.parameters())
        print(arm, "logits", tuple(y.shape), "params", n, "attn", m.attn)
