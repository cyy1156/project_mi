"""方案 21 · J1 同窗块掩码 JEPA。"""
from __future__ import annotations

import torch

from feat_index import feat_index

VIS_RAW_PTS = 600
DEFAULT_BLOCK_PTS = 50
DEFAULT_N_BLOCKS = 4


def sample_block_starts(
    batch_size: int,
    *,
    n_blocks: int = DEFAULT_N_BLOCKS,
    block_pts: int = DEFAULT_BLOCK_PTS,
    vis_pts: int = VIS_RAW_PTS,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """非重叠块起点 (B, K)，raw 点索引 ∈ [0, vis_pts - block_pts]。"""
    span = vis_pts - block_pts
    if span <= 0 or n_blocks * block_pts > vis_pts:
        raise ValueError("块配置超出可见段")
    starts = torch.empty(batch_size, n_blocks, dtype=torch.long, device=device)
    for b in range(batch_size):
        g = generator
        if g is not None:
            g = torch.Generator(device="cpu").manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()))
        perm = torch.randperm(span // block_pts + 1, generator=g)[:n_blocks]
        # 格点化非重叠：块 i 占 [perm[i]*block, perm[i]*block+block)
        used: set[int] = set()
        chosen: list[int] = []
        for _ in range(n_blocks):
            for _try in range(64):
                s = int(torch.randint(0, span, (1,), generator=g).item())
                slot = s // block_pts
                if slot not in used:
                    used.add(slot)
                    chosen.append(s)
                    break
            else:
                # fallback：顺序格点
                s = (len(chosen) * block_pts) % span
                chosen.append(s)
        starts[b] = torch.tensor(chosen[:n_blocks], dtype=torch.long)
    return starts


def apply_raw_block_mask(x: torch.Tensor, starts: torch.Tensor, block_pts: int) -> torch.Tensor:
    """x: (B,C,T) · starts: (B,K) → 被遮块置零。"""
    xm = x.clone()
    b, _c, _t = xm.shape
    for bi in range(b):
        for k in range(starts.size(1)):
            s = int(starts[bi, k].item())
            e = min(s + block_pts, xm.size(-1))
            xm[bi, :, s:e] = 0
    return xm


def block_starts_to_feat_tokens(
    starts: torch.Tensor,
    block_pts: int,
    *,
    n_times: int,
    t_prime: int,
) -> list[list[int]]:
    """每块 raw [s,s+block) → 特征 token 索引列表（每块一个 mean pool token）。"""
    out: list[list[int]] = []
    for k in range(starts.size(1)):
        s = int(starts[0, k].item())
        e = min(s + block_pts, VIS_RAW_PTS)
        i0 = feat_index(s, n_times, t_prime)
        i1 = feat_index(max(e - 1, s), n_times, t_prime)
        if i1 < i0:
            i1 = i0
        out.append(list(range(i0, i1 + 1)))
    return out


def pool_block_targets(
    feat_f: torch.Tensor,
    starts: torch.Tensor,
    *,
    n_times: int,
    t_prime: int,
    block_pts: int = DEFAULT_BLOCK_PTS,
) -> torch.Tensor:
    """feat_f: (B,D,T') · starts: (B,K) → (B,K,D) 每块 target token。"""
    feat_f = feat_f.squeeze(-1) if feat_f.ndim == 4 else feat_f
    b, d, _tp = feat_f.shape
    k = starts.size(1)
    tgt = feat_f.new_zeros(b, k, d)
    for bi in range(b):
        for j in range(k):
            s = int(starts[bi, j].item())
            e = min(s + block_pts, VIS_RAW_PTS)
            i0 = feat_index(s, n_times, t_prime)
            i1 = feat_index(max(e - 1, s), n_times, t_prime)
            i1 = min(i1, _tp - 1)
            tgt[bi, j, :] = feat_f[bi, :, i0 : i1 + 1].mean(dim=-1)
    return tgt
