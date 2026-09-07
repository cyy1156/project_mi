"""前置通道注意力：对 (B, C, T) 的 C 维重加权，输出同形。

SE  — Squeeze-and-Excitation (Hu et al., CVPR 2018)
ECA — Efficient Channel Attention (Wang et al., CVPR 2020)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _as_bct(x: torch.Tensor) -> tuple[torch.Tensor, str]:
    """统一为 (B,C,T)；返回 (tensor, layout)。"""
    if x.ndim == 3:
        return x, "bct"
    if x.ndim == 4:
        # braindecode 偶见 (B,C,T,1)
        if x.shape[-1] == 1:
            return x.squeeze(-1), "bct1"
        # (B,1,C,T)
        if x.shape[1] == 1:
            return x.squeeze(1), "b1ct"
    raise ValueError(f"SE/ECA expect (B,C,T) or (B,C,T,1)/(B,1,C,T); got {tuple(x.shape)}")


def _restore(y: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "bct":
        return y
    if layout == "bct1":
        return y.unsqueeze(-1)
    if layout == "b1ct":
        return y.unsqueeze(1)
    raise ValueError(layout)


class SE1d(nn.Module):
    """GAP over T → FC↓ → ReLU → FC↑ → Sigmoid → 广播乘回。"""

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        c = int(channels)
        r = max(int(reduction), 1)
        hidden = max(c // r, 1)
        self.fc = nn.Sequential(
            nn.Linear(c, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c, bias=False),
        )
        # 末层近零 → sigmoid(0)=0.5，配合 *2 初值近恒等
        nn.init.zeros_(self.fc[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bct, layout = _as_bct(x)
        s = x_bct.mean(dim=-1)  # (B, C)
        gate = torch.sigmoid(self.fc(s)) * 2.0  # (B, C)；零初始化 → ≈1
        y = x_bct * gate.unsqueeze(-1)
        return _restore(y, layout)


def eca_kernel_size(channels: int, gamma: float = 2.0, b: float = 1.0) -> int:
    """论文自适应核长，强制奇数且 ≥3。"""
    t = int(abs(math.log2(max(channels, 1)) / gamma + b / gamma))
    k = t if t % 2 else t + 1
    return max(k, 3)


class ECA1d(nn.Module):
    """GAP over T → 1D Conv across channels → Sigmoid → 广播乘回。"""

    def __init__(self, channels: int, k_size: int | None = 3):
        super().__init__()
        k = int(k_size) if k_size is not None else eca_kernel_size(channels)
        if k % 2 == 0:
            k += 1
        self.k_size = k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bct, layout = _as_bct(x)
        # (B, C) → (B, 1, C) for channel-wise 1D conv
        y = x_bct.mean(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, C)
        gate = torch.sigmoid(self.conv(y)).transpose(1, 2) * 2.0  # (B, C, 1)
        out = x_bct * gate
        return _restore(out, layout)
