"""LeJEPA 风格 EEG 骨干：方案 B token/mask；无 EMA；双 mask 视图 + 预测 + SIGReg。"""

from __future__ import annotations

import torch
import torch.nn as nn

from sigreg import SIGReg

N_CHANS = 8


def make_scheme_b_mask(
    batch: int,
    n_ch: int,
    n_time: int,
    mask_ratio: float,
    n_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """True = masked。随机 n_blocks 时空矩形，并集≈mask_ratio；禁止同 t 灭满全部通道。

    在 CPU/numpy 上生成后一次性拷到 device，避免逐步 .item() 同步拖慢 GPU。
    """
    import numpy as np

    n_tokens = n_ch * n_time
    target = max(1, int(round(n_tokens * float(mask_ratio))))
    n_blocks = max(1, int(n_blocks))
    out = np.zeros((batch, n_ch, n_time), dtype=bool)
    rng = np.random.default_rng()

    for b in range(batch):
        grid = np.zeros((n_ch, n_time), dtype=bool)
        base = max(1, target // n_blocks)
        sizes = [base] * (n_blocks - 1) + [max(1, target - base * (n_blocks - 1))]

        for need in sizes:
            h = int(rng.integers(1, min(4, n_ch + 1)))
            w_ideal = max(2, (int(need) + h - 1) // h)
            w = min(6, max(2, w_ideal), n_time)
            ch0 = int(rng.integers(0, n_ch - h + 1))
            t0 = int(rng.integers(0, n_time - w + 1))
            grid[ch0 : ch0 + h, t0 : t0 + w] = True
            for t in range(t0, t0 + w):
                if grid[:, t].all():
                    keep = ch0 + int(rng.integers(0, h))
                    grid[keep, t] = False

        flat = grid.reshape(-1)
        cur = int(flat.sum())
        if cur > target:
            idx = np.flatnonzero(flat)
            drop = rng.choice(idx, size=cur - target, replace=False)
            flat[drop] = False
        elif cur < target:
            free = np.flatnonzero(~flat)
            rng.shuffle(free)
            for i in free:
                if int(flat.sum()) >= target:
                    break
                c, t = divmod(int(i), n_time)
                if int(grid[:, t].sum()) >= n_ch - 1:
                    continue
                flat[i] = True
                grid[c, t] = True

        if not flat.any():
            flat[0] = True
        grid = flat.reshape(n_ch, n_time)
        full_cols = np.where(grid.all(axis=0))[0]
        for t in full_cols:
            grid[int(rng.integers(0, n_ch)), t] = False
        out[b] = grid

    return torch.from_numpy(out.reshape(batch, n_tokens)).to(device=device, non_blocking=True)

class PatchEmbed(nn.Module):
    """(B,8,T) → (B, 8 * n_time, D)；与 10 号同构。"""

    def __init__(self, n_times: int, patch_time: int, embed_dim: int, n_chans: int = N_CHANS):
        super().__init__()
        assert n_times % patch_time == 0
        self.patch_time = int(patch_time)
        self.n_time = n_times // patch_time
        self.n_spat = int(n_chans)
        self.n_tokens = self.n_spat * self.n_time
        self.proj = nn.Sequential(
            nn.Linear(self.patch_time, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        assert c == self.n_spat and t == self.n_time * self.patch_time
        patches = x.reshape(b, c, self.n_time, self.patch_time)
        tok = self.proj(patches)
        return tok.reshape(b, self.n_tokens, -1) + self.pos


class TokenEncoder(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, drop: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(tokens))


class EEGLeJEPA(nn.Module):
    """单编码器 · 双视图预测 · SIGReg（无 teacher-student）。"""

    def __init__(
        self,
        n_times: int = 500,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        patch_time: int = 25,
        drop_prob: float = 0.1,
        mask_ratio: float = 0.25,
        mask_n_blocks: int = 4,
        lambda_sigreg: float = 0.05,
        num_slices: int = 256,
    ):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.mask_n_blocks = int(mask_n_blocks)
        self.lambda_sigreg = float(lambda_sigreg)
        self.embed = PatchEmbed(n_times, patch_time, embed_dim)
        self.encoder = TokenEncoder(embed_dim, n_heads, n_layers, drop=drop_prob)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sigreg = SIGReg(num_slices=num_slices, num_points=17)
        self.n_tokens = self.embed.n_tokens
        self.n_spat = self.embed.n_spat
        self.n_time = self.embed.n_time
        self.embed_dim = embed_dim

    def _make_mask(self, batch: int, device: torch.device) -> torch.Tensor:
        return make_scheme_b_mask(
            batch,
            self.n_spat,
            self.n_time,
            self.mask_ratio,
            self.mask_n_blocks,
            device,
        )

    def encode_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tok = self.embed(x)
        mt = self.mask_token.expand_as(tok)
        tok = torch.where(mask.unsqueeze(-1), mt, tok)
        return self.encoder(tok)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.squeeze(1)
        return self.encoder(self.embed(x)).mean(dim=1)

    def forward_lejepa(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if x.ndim == 4:
            x = x.squeeze(1)
        m1 = self._make_mask(x.size(0), x.device)
        m2 = self._make_mask(x.size(0), x.device)
        h1 = self.encode_masked(x, m1)
        h2 = self.encode_masked(x, m2)
        z1 = h1.mean(dim=1)
        z2 = h2.mean(dim=1)
        p12 = self.predictor(z1)
        p21 = self.predictor(z2)
        l_pred = 0.5 * ((p12 - z2).pow(2).mean() + (p21 - z1).pow(2).mean())
        z_all = torch.cat([z1, z2], dim=0)
        l_sig = self.sigreg(z_all)
        loss = l_pred + self.lambda_sigreg * l_sig
        info = {
            "l_pred": float(l_pred.detach()),
            "l_sigreg": float(l_sig.detach()),
            "z_std": float(z_all.detach().std()),
            "mask_frac": float(0.5 * (m1.float().mean() + m2.float().mean())),
        }
        return loss, info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class LeJepaClassifier(nn.Module):
    def __init__(self, backbone: EEGLeJEPA, n_outputs: int = 3, drop_prob: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(backbone.embed_dim, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.pool(x))


def freeze_backbone(model: LeJepaClassifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: LeJepaClassifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = True
