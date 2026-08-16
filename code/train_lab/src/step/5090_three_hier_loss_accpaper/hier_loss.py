"""Scheme 16 · hierarchical / LR-aware losses for Three (3 logits).

Labels: 0=idle, 1=left, 2=right.
Task (n_outputs=2) always falls back to plain CE.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 训练环会临时 monkey-patch nn.CrossEntropyLoss；此处保留真实类供 S0 / 子 CE 使用。
_CrossEntropyLoss = nn.CrossEntropyLoss


@dataclass(frozen=True)
class HierLossHP:
    arm: str = "H1"
    lambda_ce: float = 1.0
    lambda_mi: float = 0.5
    lambda_lr: float = 0.75
    lambda_mg: float = 0.3
    lambda_id: float = 0.2
    lambda_tr: float = 0.0  # H3 only; needs trial ids in batch — off by default
    margin: float = 1.0


# Presets: which terms are on
_ARM_FLAGS = {
    "S0": dict(use_mi=False, use_lr=False, use_mg=False, use_id=False),
    "H1": dict(use_mi=True, use_lr=True, use_mg=False, use_id=False),
    "H2": dict(use_mi=True, use_lr=True, use_mg=True, use_id=True),
    "H3": dict(use_mi=True, use_lr=True, use_mg=True, use_id=True),  # + trial later
}


class HierThreeLoss(nn.Module):
    def __init__(self, hp: HierLossHP):
        super().__init__()
        self.hp = hp
        flags = _ARM_FLAGS.get(hp.arm, _ARM_FLAGS["H1"])
        self.use_mi = flags["use_mi"]
        self.use_lr = flags["use_lr"]
        self.use_mg = flags["use_mg"]
        self.use_id = flags["use_id"]
        self.ce = _CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        if logits.shape[-1] != 3:
            # Task binary or unexpected → plain CE
            return self.ce(logits, y)

        hp = self.hp
        loss = hp.lambda_ce * self.ce(logits, y)

        if self.use_mi:
            y_mi = (y != 0).long()
            z_idle = logits[:, 0:1]
            z_mi = torch.logsumexp(logits[:, 1:3], dim=1, keepdim=True)
            loss = loss + hp.lambda_mi * self.ce(torch.cat([z_idle, z_mi], dim=1), y_mi)

        mi = y > 0
        if self.use_lr and mi.any():
            y_lr = (y[mi] - 1).long()  # 1→0 left, 2→1 right
            loss = loss + hp.lambda_lr * self.ce(logits[mi][:, 1:3], y_lr)

        if self.use_mg and mi.any():
            z = logits[mi]
            y_mi_cls = y[mi]
            z_pos = torch.where(y_mi_cls == 1, z[:, 1], z[:, 2])
            z_neg = torch.where(y_mi_cls == 1, z[:, 2], z[:, 1])
            loss = loss + hp.lambda_mg * F.relu(hp.margin - (z_pos - z_neg)).mean()

        if self.use_id:
            p = F.softmax(logits, dim=1)
            idle = y == 0
            term = torch.zeros((), device=logits.device, dtype=logits.dtype)
            n = 0
            if idle.any():
                term = term + p[idle][:, 1:3].sum(dim=1).mean()
                n += 1
            if mi.any():
                term = term + p[mi][:, 0].mean()
                n += 1
            if n:
                loss = loss + hp.lambda_id * (term / n)

        return loss


def build_criterion(arm: str, n_outputs: int, hp: HierLossHP | None = None) -> nn.Module:
    if n_outputs != 3 or arm == "S0":
        return _CrossEntropyLoss()
    cfg = hp or HierLossHP(arm=arm)
    if arm != cfg.arm:
        cfg = HierLossHP(
            arm=arm,
            lambda_ce=cfg.lambda_ce,
            lambda_mi=cfg.lambda_mi,
            lambda_lr=cfg.lambda_lr,
            lambda_mg=cfg.lambda_mg,
            lambda_id=cfg.lambda_id,
            lambda_tr=cfg.lambda_tr,
            margin=cfg.margin,
        )
    return HierThreeLoss(cfg)


def loss_meta(arm: str, hp: HierLossHP | None = None) -> dict:
    cfg = hp or HierLossHP(arm=arm)
    return {
        "scheme": "16",
        "arm": arm,
        "loss": arm,
        "hier_loss": asdict(cfg),
        "backbone": "braindecode.ShallowFBCSPNet",
        "accpaper": True,
        "bypass": True,
    }
