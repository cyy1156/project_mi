"""SIGReg（论文 LeJEPA 精简实现）：随机切片 + Epps–Pulley 特征函数匹配 N(0,1)。

官方 pip 包 `lejepa` 本机不可用时用本实现；若日后 `pip install lejepa` 成功，
可在 shared_hparams 里切 use_official_lejepa=True（预留）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization（可微、线性复杂度）。"""

    def __init__(self, num_slices: int = 256, num_points: int = 17):
        super().__init__()
        self.num_slices = int(num_slices)
        # Epps–Pulley 常用频率点（覆盖 CF）
        omegas = torch.linspace(0.05, 3.0, steps=int(num_points))
        self.register_buffer("omegas", omegas, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (N, D) embeddings
        """
        if z.ndim != 2:
            raise ValueError(f"SIGReg expects (N,D), got {tuple(z.shape)}")
        n, d = z.shape
        if n < 2:
            return z.new_zeros(())
        # 中心化（各向同性高斯的平移不变部分）
        z = z - z.mean(dim=0, keepdim=True)
        # 随机单位投影
        proj = torch.randn(d, self.num_slices, device=z.device, dtype=z.dtype)
        proj = F.normalize(proj, dim=0)
        s = z @ proj  # (N, S)
        # 逐切片标准化到经验方差，再对齐 N(0,1) 的 CF（稳定版）
        s = s / (s.std(dim=0, keepdim=True) + 1e-5)
        # Epps–Pulley：E[cos(ωs)]→e^{-ω²/2}, E[sin]→0
        loss = z.new_zeros(())
        for w in self.omegas.tolist():
            w = float(w)
            cos_m = torch.cos(w * s).mean(dim=0)
            sin_m = torch.sin(w * s).mean(dim=0)
            target = math.exp(-0.5 * w * w)
            loss = loss + ((cos_m - target).pow(2) + sin_m.pow(2)).mean()
        loss = loss / float(len(self.omegas))
        return loss
