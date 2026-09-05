"""最小 EEG-JEPA：方案 B（8ch×时间块 token）+ EMA target + 四块 25% mask。"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

# OpenBMI hop100 通道序：Cz, C3, C4, CP3, FC4, FC3, CP4, CPz
N_CHANS = 8


def make_scheme_b_mask(
    batch: int,
    n_ch: int,
    n_time: int,
    mask_ratio: float,
    n_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """True = masked。随机 n_blocks 时空矩形，并集≈mask_ratio；禁止同 t 灭满全部通道。"""
    n_tokens = n_ch * n_time
    target = max(1, int(round(n_tokens * float(mask_ratio))))
    n_blocks = max(1, int(n_blocks))
    mask = torch.zeros(batch, n_tokens, dtype=torch.bool, device=device)

    for b in range(batch):
        grid = torch.zeros(n_ch, n_time, dtype=torch.bool, device=device)
        base = max(1, target // n_blocks)
        sizes = [base] * (n_blocks - 1) + [max(1, target - base * (n_blocks - 1))]

        for need in sizes:
            h = int(torch.randint(1, min(4, n_ch + 1), (1,), device=device).item())
            w_ideal = max(2, (int(need) + h - 1) // h)
            w = min(6, max(2, w_ideal), n_time)
            ch0 = int(torch.randint(0, n_ch - h + 1, (1,), device=device).item())
            t0 = int(torch.randint(0, n_time - w + 1, (1,), device=device).item())
            grid[ch0 : ch0 + h, t0 : t0 + w] = True
            for t in range(t0, t0 + w):
                if bool(grid[:, t].all()):
                    # 优先在本块通道内留一可见
                    keep = ch0 + int(torch.randint(0, h, (1,), device=device).item())
                    grid[keep, t] = False

        flat = grid.reshape(-1)
        cur = int(flat.sum().item())
        if cur > target:
            idx = flat.nonzero(as_tuple=False).flatten()
            drop = idx[torch.randperm(idx.numel(), device=device)[: cur - target]]
            flat[drop] = False
        elif cur < target:
            free = (~flat).nonzero(as_tuple=False).flatten()
            free = free[torch.randperm(free.numel(), device=device)]
            for i in free.tolist():
                if int(flat.sum().item()) >= target:
                    break
                c, t = divmod(int(i), n_time)
                if int(grid[:, t].sum().item()) >= n_ch - 1:
                    continue
                flat[i] = True
                grid[c, t] = True

        if not bool(flat.any()):
            flat[0] = True
        # 再扫一遍列约束（补点后可能触线）
        grid = flat.view(n_ch, n_time)
        for t in range(n_time):
            if bool(grid[:, t].all()):
                keep = int(torch.randint(0, n_ch, (1,), device=device).item())
                grid[keep, t] = False
        mask[b] = grid.reshape(-1)
    return mask


class PatchEmbed(nn.Module):
    """(B,8,T) → (B, 8 * n_time, D)；逐通道 100ms token，下标 ch*n_time+t。"""

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
        # (B, C, n_time, patch) → Linear on last dim
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

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        h = self.blocks(tokens, src_key_padding_mask=key_padding_mask)
        return self.norm(h)


class Predictor(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.blocks(tokens))


class EEGJepa(nn.Module):
    """窗内 JEPA；pool() 供下游线性头。"""

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
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.mask_n_blocks = int(mask_n_blocks)
        self.ema_momentum = float(ema_momentum)
        self.embed = PatchEmbed(n_times, patch_time, embed_dim)
        self.context = TokenEncoder(embed_dim, n_heads, n_layers, drop=drop_prob)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = Predictor(embed_dim, n_heads=max(1, n_heads // 2), n_layers=1)
        self.target_embed = copy.deepcopy(self.embed)
        self.target_context = copy.deepcopy(self.context)
        for p in list(self.target_embed.parameters()) + list(self.target_context.parameters()):
            p.requires_grad = False
        self.n_tokens = self.embed.n_tokens
        self.n_spat = self.embed.n_spat
        self.n_time = self.embed.n_time
        self.embed_dim = embed_dim

    @torch.no_grad()
    def update_ema(self) -> None:
        m = self.ema_momentum
        pairs = (
            (self.embed, self.target_embed),
            (self.context, self.target_context),
        )
        for src, tgt in pairs:
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

    def _make_mask(self, batch: int, device: torch.device) -> torch.Tensor:
        return make_scheme_b_mask(
            batch,
            self.n_spat,
            self.n_time,
            self.mask_ratio,
            self.mask_n_blocks,
            device,
        )

    def encode_context(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tok = self.embed(x)
        mt = self.mask_token.expand(tok.size(0), tok.size(1), -1)
        tok = torch.where(mask.unsqueeze(-1), mt, tok)
        return self.context(tok)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.target_embed(x)
        return self.target_context(tok)

    def forward_jepa(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if x.ndim == 4:
            x = x.squeeze(1)
        mask = self._make_mask(x.size(0), x.device)
        ctx = self.encode_context(x, mask)
        pred = self.predictor(ctx)
        with torch.no_grad():
            tgt = self.encode_target(x)
        diff = (pred - tgt)[mask]
        loss = diff.pow(2).mean() if diff.numel() else pred.new_zeros(())
        info = {
            "mask_frac": float(mask.float().mean().item()),
            "pred_std": float(pred.detach().std().item()),
            "tgt_std": float(tgt.detach().std().item()),
        }
        return loss, info

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.squeeze(1)
        tok = self.embed(x)
        h = self.context(tok)
        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class JepaClassifier(nn.Module):
    def __init__(self, backbone: EEGJepa, n_outputs: int = 3, drop_prob: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(backbone.embed_dim, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.pool(x))


def freeze_backbone(model: JepaClassifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: JepaClassifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in list(model.backbone.target_embed.parameters()) + list(
        model.backbone.target_context.parameters()
    ):
        p.requires_grad = False
