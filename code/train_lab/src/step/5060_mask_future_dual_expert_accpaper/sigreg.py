"""SIGReg · 与 5060_lejepa_three_probe 同实现；开跑 num_slices=1024。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    def __init__(self, num_slices: int = 1024, num_points: int = 17):
        super().__init__()
        self.num_slices = int(num_slices)
        omegas = torch.linspace(0.05, 3.0, steps=int(num_points))
        self.register_buffer("omegas", omegas, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"SIGReg expects (N,D), got {tuple(z.shape)}")
        n, d = z.shape
        if n < 2:
            return z.new_zeros(())
        z = z - z.mean(dim=0, keepdim=True)
        proj = torch.randn(d, self.num_slices, device=z.device, dtype=z.dtype)
        proj = F.normalize(proj, dim=0)
        s = z @ proj
        s = s / (s.std(dim=0, keepdim=True) + 1e-5)
        loss = z.new_zeros(())
        for w in self.omegas.tolist():
            w = float(w)
            cos_m = torch.cos(w * s).mean(dim=0)
            sin_m = torch.sin(w * s).mean(dim=0)
            target = math.exp(-0.5 * w * w)
            loss = loss + ((cos_m - target).pow(2) + sin_m.pow(2)).mean()
        return loss / float(len(self.omegas))
