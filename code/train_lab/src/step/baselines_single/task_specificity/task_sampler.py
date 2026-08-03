"""Task 训练采样：类别平衡（inverse-frequency sample weights）。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(
    y_train: np.ndarray,
    *,
    n_classes: int = 2,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """按类频数逆权重抽样，使长期 batch 内两类接近 1:1。

    - 每样本权重 = 1 / n_c（c 为该样本类别）
    - num_samples = len(y_train)，有放回
    - 与 DataLoader(shuffle=True) 互斥：有 sampler 时不要再 shuffle
    """
    y = np.asarray(y_train).astype(int).reshape(-1)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    sample_w = 1.0 / counts[y]
    weights = torch.as_tensor(sample_w, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(len(y)),
        replacement=True,
        generator=generator,
    )
