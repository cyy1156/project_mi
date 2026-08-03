"""Task 二分类训练目标（特异度目录专用）。

- 加权交叉熵：fixed / invfreq
- Focal Loss：压低易样本权重，可选类 α（加重静息）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_task_weights(w0: float = 2.0, w1: float = 1.0) -> torch.Tensor:
    """返回 shape (2,)：index0=静息, index1=任务。"""
    return torch.tensor([float(w0), float(w1)], dtype=torch.float32)


def inverse_freq_task_weights(y_train: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    """逆频率：w_c = N / (C * n_c)，再归一化到均值 1。"""
    y = np.asarray(y_train).astype(int).reshape(-1)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n_total = float(counts.sum())
    w = n_total / (n_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def build_task_ce(
    device: torch.device,
    *,
    mode: str = "fixed",
    w0: float = 2.0,
    w1: float = 1.0,
    y_train: np.ndarray | None = None,
) -> nn.CrossEntropyLoss:
    """
    mode:
      - "fixed": 使用 w0/w1
      - "invfreq": 使用 y_train 逆频率（需传入该折 train 标签）
    """
    if mode == "fixed":
        weight = fixed_task_weights(w0=w0, w1=w1)
    elif mode == "invfreq":
        if y_train is None:
            raise ValueError("mode='invfreq' 需要 y_train")
        weight = inverse_freq_task_weights(y_train)
    else:
        raise ValueError(f"未知 mode={mode}")
    return nn.CrossEntropyLoss(weight=weight.to(device))


class FocalLoss(nn.Module):
    """二分类/多分类 Focal Loss：FL = -α_t (1-p_t)^γ log(p_t)。"""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", alpha.detach().float().clone())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        idx = target.long().unsqueeze(1)
        log_pt = log_probs.gather(1, idx).squeeze(1)
        pt = probs.gather(1, idx).squeeze(1)
        loss = -(1.0 - pt).pow(self.gamma) * log_pt
        if self.alpha is not None:
            at = self.alpha.to(logits.device)[target.long()]
            loss = loss * at
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_task_focal(
    device: torch.device,
    *,
    gamma: float = 2.0,
    alpha0: float = 0.75,
    alpha1: float = 0.25,
    use_alpha: bool = True,
) -> FocalLoss:
    """
    默认 γ=2；α0=0.75（静息）、α1=0.25（任务），与少数类静息对齐。
    use_alpha=False 时仅用 focusing 项。
    """
    alpha = None
    if use_alpha:
        alpha = torch.tensor([float(alpha0), float(alpha1)], dtype=torch.float32)
    return FocalLoss(gamma=gamma, alpha=alpha).to(device)
