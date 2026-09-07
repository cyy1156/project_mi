"""FastKAN · 1 隐层 · 纯 torch（方案 26 E2a）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GaussianRBFLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)
        centers = torch.linspace(-1.5, 1.5, grid_size)
        self.register_buffer("centers", centers)
        self.log_sigma = nn.Parameter(torch.zeros(out_dim, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        x = x.unsqueeze(1).unsqueeze(-1)  # B,1,in,1
        c = self.centers.view(1, 1, 1, -1)
        sigma = torch.exp(self.log_sigma).view(1, self.out_dim, self.in_dim, 1)
        dist = (x - c) ** 2
        basis = torch.exp(-dist / (2 * sigma ** 2 + 1e-6))
        w = self.coeff.unsqueeze(0)
        out = (basis * w).sum(dim=(-2, -1))
        return out


class FastKANClassifier(nn.Module):
    def __init__(self, in_dim: int, n_outputs: int, hidden: int = 64, grid: int = 5):
        super().__init__()
        self.rbf = GaussianRBFLayer(in_dim, hidden, grid_size=grid)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        h = self.rbf(x)
        h = torch.tanh(h)
        return self.head(h)
