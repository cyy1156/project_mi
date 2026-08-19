"""Future token 的 Phase 查表（metadata · 无 Future EEG 泄漏）。

Phase id（MI trial）：
  0 = onset   · t_rel_cue < 0.5s
  1 = sustain · 0.5s ≤ t_rel_cue < 3.5s
  2 = offset  · t_rel_cue ≥ 3.5s（含 post-MI future）
  3 = rest    · y_three=0（Rest/idle；Future 仍在 Cue 前）

在线推理：仅用 t0_sec + y 的 trial 类型（Rest 固定规则）+ 窗几何，不读 Future 波形。
"""
from __future__ import annotations

import torch

from feat_index import feat_index

MI_SEC = 4.0
ONSET_END_SEC = 0.5
OFFSET_START_SEC = MI_SEC - 0.5  # 3.5
N_PHASE = 4  # onset, sustain, offset, rest


def sample_index_from_feat(fi: int, *, n_times: int = 1000, t_prime: int = 61) -> float:
    """特征时间索引 → 原始采样点（连续）。"""
    return float(fi) / float(t_prime - 1) * float(n_times - 1)


def time_rel_cue_from_feat(
    t0_sec: torch.Tensor | float,
    fi: int,
    *,
    n_times: int = 1000,
    t_prime: int = 61,
) -> torch.Tensor:
    """窗内 future token 中心相对 cue 的时间（秒）。"""
    s = sample_index_from_feat(fi, n_times=n_times, t_prime=t_prime)
    # 采样点 100 = current 段起点 = t0_sec（相对 cue）
    if isinstance(t0_sec, torch.Tensor):
        return t0_sec + (s - 100.0) / 250.0
    return torch.as_tensor(t0_sec, dtype=torch.float32) + (s - 100.0) / 250.0


def _phase_scalar(t_rel: float, y: int) -> int:
    if y == 0:
        return 3
    if t_rel < ONSET_END_SEC:
        return 0
    if t_rel < OFFSET_START_SEC:
        return 1
    return 2


def future_phase_ids(
    t0_sec: torch.Tensor,
    y: torch.Tensor,
    i_fut: list[int],
    *,
    n_times: int = 1000,
    t_prime: int = 61,
) -> torch.Tensor:
    """(B, L_fut) long — 每个 future token 的 phase id。"""
    b = int(t0_sec.size(0))
    lf = len(i_fut)
    out = torch.zeros(b, lf, dtype=torch.long, device=t0_sec.device)
    for j, fi in enumerate(i_fut):
        t_rel = time_rel_cue_from_feat(t0_sec, fi, n_times=n_times, t_prime=t_prime)
        for i in range(b):
            out[i, j] = _phase_scalar(float(t_rel[i].item()), int(y[i].item()))
    return out


def future_phase_ids_fast(
    t0_sec: torch.Tensor,
    y: torch.Tensor,
    i_fut: list[int],
    *,
    n_times: int = 1000,
    t_prime: int = 61,
) -> torch.Tensor:
    """向量化版 future phase（训练热路径）。"""
    fi = torch.tensor(i_fut, device=t0_sec.device, dtype=torch.float32)
    s = fi / float(t_prime - 1) * float(n_times - 1)
    t_rel = t0_sec.unsqueeze(1) + (s.unsqueeze(0) - 100.0) / 250.0  # (B, L_fut)
    yv = y.view(-1, 1).expand_as(t_rel)
    out = torch.full_like(t_rel, 2, dtype=torch.long)  # default offset
    rest = yv == 0
    out = torch.where(rest, torch.full_like(out, 3), out)
    mi = ~rest
    out = torch.where(mi & (t_rel < ONSET_END_SEC), torch.zeros_like(out), out)
    out = torch.where(
        mi & (t_rel >= ONSET_END_SEC) & (t_rel < OFFSET_START_SEC),
        torch.ones_like(out),
        out,
    )
    return out


def default_i_fut(t_prime: int = 61, n_times: int = 1000) -> list[int]:
    i_split = feat_index(600, n_times, t_prime)
    return list(range(i_split, t_prime))
