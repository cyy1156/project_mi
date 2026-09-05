"""CIACNet (Liao et al. 2025) — PyTorch reimplementation for P/L tracks."""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


Ablation = Literal["full", "no_cv2", "no_iat", "no_tc", "std_cbam"]


def _same_pad_1d(k: int) -> tuple[int, int]:
    """Symmetric pad for kernel k (odd/even)."""
    left = (k - 1) // 2
    right = k - 1 - left
    return left, right


class ConvBlock(nn.Module):
    """EEGNet-like branch: temporal → depthwise spatial → pool → temporal → pool."""

    def __init__(
        self,
        n_chans: int,
        f_time: int,
        kc: int,
        f_out: int,
        depth: int = 2,
        dropout: float = 0.3,
        conv_l2: float = 1e-4,
    ):
        super().__init__()
        self.conv_l2 = conv_l2
        # Input: (B, 1, C, T) — same padding keeps T stable before pools (/8,/8)
        self.temp1 = nn.Conv2d(1, f_time, kernel_size=(1, kc), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f_time)
        self.depth = nn.Conv2d(
            f_time, f_time * depth, kernel_size=(n_chans, 1), groups=f_time, bias=False
        )
        self.bn2 = nn.BatchNorm2d(f_time * depth)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop1 = nn.Dropout(dropout)
        self.temp2 = nn.Conv2d(
            f_time * depth, f_out, kernel_size=(1, 16), padding="same", bias=False
        )
        self.bn3 = nn.BatchNorm2d(f_out)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.elu(self.bn1(self.temp1(x)))
        x = self.elu(self.bn2(self.depth(x)))
        x = self.drop1(self.pool1(x))
        x = self.elu(self.bn3(self.temp2(x)))
        x = self.drop2(self.pool2(x))
        return x

    def l2_penalty(self) -> torch.Tensor:
        regs = [self.temp1.weight, self.depth.weight, self.temp2.weight]
        return self.conv_l2 * sum(w.pow(2).sum() for w in regs)


def _sto_sample_or_expect(values: torch.Tensor, train: bool) -> torch.Tensor:
    """values: (..., K). Prob ∝ softplus(a); train sample, eval E[a·p]."""
    # softplus keeps gradient & avoids all-zero after ReLU clamp on early random weights
    weights = F.softplus(values) + 1e-8
    sums = weights.sum(dim=-1, keepdim=True)
    probs = weights / sums
    if train:
        flat = probs.reshape(-1, values.size(-1))
        idx = torch.multinomial(flat, 1)
        gathered = values.reshape(-1, values.size(-1)).gather(-1, idx)
        return gathered.reshape(*values.shape[:-1], 1)
    # expectation under sampling distribution
    return (values * probs).sum(dim=-1, keepdim=True)


def stochastic_pool_global(x: torch.Tensor, train: bool) -> torch.Tensor:
    """Global stochastic pool over H×W → (B,C,1,1)."""
    b, c, h, w = x.shape
    flat = x.reshape(b, c, h * w)
    out = _sto_sample_or_expect(flat, train)
    return out.reshape(b, c, 1, 1)


def stochastic_pool_channel(x: torch.Tensor, train: bool) -> torch.Tensor:
    """Stochastic pool over channel dim → (B,1,H,W)."""
    b, c, h, w = x.shape
    flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
    out = _sto_sample_or_expect(flat, train)
    return out.reshape(b, h, w, 1).permute(0, 3, 1, 2)


class ImprovedCBAM(nn.Module):
    """IAT: Avg+Max+Stochastic channel & spatial attention (r=8, spatial k=7)."""

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7, use_sto: bool = True):
        super().__init__()
        self.use_sto = use_sto
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        pad = spatial_kernel // 2
        in_ch = 3 if use_sto else 2
        self.spatial = nn.Conv2d(in_ch, 1, kernel_size=spatial_kernel, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        parts = [avg, mx]
        if self.use_sto:
            parts.append(stochastic_pool_global(x, self.training))
        # MLP on each descriptor then sum (paper: aligned positions summed)
        att = 0
        for p in parts:
            att = att + self.mlp(p.view(b, c))
        mc = torch.sigmoid(att).view(b, c, 1, 1)
        x = x * mc

        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.amax(dim=1, keepdim=True)
        spat_in = [avg_s, max_s]
        if self.use_sto:
            spat_in.append(stochastic_pool_channel(x, self.training))
        ms = torch.sigmoid(self.spatial(torch.cat(spat_in, dim=1)))
        return x * ms


class TCNResidual(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # causal crop
        y = self.conv1(x)
        y = y[..., : x.size(-1)]
        y = self.drop1(self.relu(y))
        y = self.conv2(y)
        y = y[..., : x.size(-1)]
        y = self.drop2(self.relu(y))
        return self.relu(x + y)


class TCBlock(nn.Module):
    """TCN L=2, KT=4, F=32 on flattened IAT feature sequence."""

    def __init__(self, in_ch: int, filters: int = 32, kernel: int = 4, n_blocks: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, filters, 1)
        blocks = []
        for i in range(n_blocks):
            blocks.append(TCNResidual(filters, kernel, dilation=2**i, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 1, T) → (B, C, T)
        if x.dim() == 4:
            x = x.squeeze(2)
        x = self.proj(x)
        return self.blocks(x)


class CIACNet(nn.Module):
    def __init__(
        self,
        n_chans: int = 8,
        n_times: int = 1125,
        n_outputs: int = 4,
        dropout: float = 0.3,
        conv_l2: float = 1e-4,
        ablation: Ablation = "full",
    ):
        super().__init__()
        self.ablation = ablation
        self.conv_l2 = conv_l2
        use_cv2 = ablation != "no_cv2"
        use_iat = ablation != "no_iat"
        use_tc = ablation != "no_tc"
        use_sto = ablation != "std_cbam"

        self.cv1 = ConvBlock(n_chans, f_time=16, kc=32, f_out=32, depth=2, dropout=dropout, conv_l2=conv_l2)
        self.cv2 = ConvBlock(n_chans, f_time=32, kc=64, f_out=64, depth=2, dropout=dropout, conv_l2=conv_l2) if use_cv2 else None
        self.iat = ImprovedCBAM(32, reduction=8, use_sto=use_sto) if use_iat else nn.Identity()
        self.use_tc = use_tc
        self.tc = TCBlock(32, filters=32, kernel=4, n_blocks=2, dropout=dropout) if use_tc else None

        # Infer flatten sizes with a dry run
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            f1 = self.cv1(dummy)
            fi = self.iat(f1)
            parts = [fi.flatten(1)]
            if self.tc is not None:
                ft = self.tc(fi)
                parts.append(ft.flatten(1))
            else:
                # still keep IAT bypass as paper's parallel branch; if no TC, only IAT (+CV2)
                pass
            if self.cv2 is not None:
                f2 = self.cv2(dummy)
                parts.append(f2.flatten(1))
            feat_dim = sum(p.shape[1] for p in parts)

        self.fc = nn.Linear(feat_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept (B,C,T) or (B,1,C,T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        f1 = self.cv1(x)
        fi = self.iat(f1)
        feats = [fi.flatten(1)]
        if self.tc is not None:
            feats.append(self.tc(fi).flatten(1))
        if self.cv2 is not None:
            feats.append(self.cv2(x).flatten(1))
        return self.fc(torch.cat(feats, dim=1))

    def loss_reg(self) -> torch.Tensor:
        reg = self.cv1.l2_penalty()
        if self.cv2 is not None:
            reg = reg + self.cv2.l2_penalty()
        return reg


def build_ciacnet(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    drop_prob: float = 0.3,
    ablation: Ablation = "full",
    conv_l2: float = 1e-4,
) -> CIACNet:
    return CIACNet(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        dropout=drop_prob,
        conv_l2=conv_l2,
        ablation=ablation,
    )
