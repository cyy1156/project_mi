"""方案 26 · 训练配方：AdamW / cosine / LS / clip / SWA。"""

from __future__ import annotations

import copy
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from s26_hparams import RecipeTrainHP


def make_criterion(hp: RecipeTrainHP) -> nn.Module:
    ls = float(getattr(hp, "label_smoothing", 0.0) or 0.0)
    if ls > 0:
        return nn.CrossEntropyLoss(label_smoothing=ls)
    return nn.CrossEntropyLoss()


def make_optimizer(model: torch.nn.Module, hp: RecipeTrainHP) -> torch.optim.Optimizer:
    opt = str(getattr(hp, "optimizer", "adam") or "adam").lower()
    if opt == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
    return torch.optim.Adam(
        model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, hp: RecipeTrainHP, *, total_epochs: int
):
    warmup = int(getattr(hp, "warmup_epochs", 0) or 0)

    def lr_lambda(ep: int) -> float:
        if ep < warmup:
            return float(ep + 1) / float(max(warmup, 1))
        progress = (ep - warmup) / float(max(total_epochs - warmup, 1))
        progress = min(max(progress, 0.0), 1.0)
        lo = hp.lr_min / hp.lr
        return lo + 0.5 * (1.0 - lo) * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def is_improved(score: float, best: float, hp: RecipeTrainHP) -> bool:
    eps = float(getattr(hp, "early_stop_tie_eps", 0.0) or 0.0)
    return score > best + eps


def maybe_clip_grad(model: torch.nn.Module, hp: RecipeTrainHP) -> None:
    norm = float(getattr(hp, "grad_clip_norm", 0.0) or 0.0)
    if norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm)


def run_epoch_recipe(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    hp: RecipeTrainHP,
    *,
    non_blocking: bool = True,
    use_amp: bool = False,
    scaler=None,
) -> float:
    model.train(train)
    total, n = 0.0, 0
    amp_on = bool(use_amp) and device.type == "cuda"
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if len(batch) == 3:
                x, y, sw = batch
                sw = sw.to(device, non_blocking=non_blocking)
            else:
                x, y = batch
                sw = None
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if amp_on:
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(x)
                    if logits.ndim > 2:
                        logits = logits.reshape(logits.shape[0], -1)
                    if sw is not None:
                        loss = nn.functional.cross_entropy(logits, y, reduction="none")
                        loss = (loss * sw).mean()
                    else:
                        loss = criterion(logits, y)
            else:
                logits = model(x)
                if logits.ndim > 2:
                    logits = logits.reshape(logits.shape[0], -1)
                if sw is not None:
                    loss = nn.functional.cross_entropy(logits, y, reduction="none")
                    loss = (loss * sw).mean()
                else:
                    loss = criterion(logits, y)
            if train:
                if amp_on and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    maybe_clip_grad(model, hp)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    maybe_clip_grad(model, hp)
                    optimizer.step()
            total += float(loss.item()) * len(y)
            n += len(y)
    return total / max(n, 1)


def setup_swa(model: torch.nn.Module, hp: RecipeTrainHP, total_epochs: int):
    if not hp.use_swa:
        return None, None
    start = max(int(total_epochs * hp.swa_start_frac), 1)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(
        torch.optim.SGD(model.parameters(), lr=hp.lr * 0.1),
        swa_lr=hp.lr * 0.05,
    )
    return swa_model, (start, swa_sched)


def finalize_swa(
    swa_model: AveragedModel | None,
    model: torch.nn.Module,
    train_loader,
    device,
) -> dict | None:
    if swa_model is None:
        return None
    update_bn(train_loader, swa_model, device=device)
    return {k: v.detach().cpu().clone() for k, v in swa_model.module.state_dict().items()}
