"""AdaBN：仅更新 ShallowFBCSPNet 内 BatchNorm running 统计。

v1.2（默认）：predict_first 严格因果 — eval 预测后用 train 更新 running stats。
v1.1：train 更新后 eval 预测（当前窗统计进入预测）。
v1.0（已弃用）：train 下单窗预测，会导致 Three 崩溃。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

ADABN_VERSION_DEFAULT = "v1.2"


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def snapshot_bn_running(model: nn.Module) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            out[name] = (mod.running_mean.detach().clone(), mod.running_var.detach().clone())
    return out


def restore_bn_running(
    model: nn.Module, snap: dict[str, tuple[torch.Tensor, torch.Tensor]]
) -> None:
    for name, mod in model.named_modules():
        if name in snap:
            rm, rv = snap[name]
            mod.running_mean.copy_(rm)
            mod.running_var.copy_(rv)


def assert_params_unchanged(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> None:
    for k in before:
        if not torch.equal(before[k], after[k]):
            raise AssertionError(f"AdaBN 改动了可训练参数: {k}")


def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


@dataclass
class StreamPred:
    pred: np.ndarray
    prob_max: np.ndarray
    latency_ms: list[float]


def _window_batch_tensor(w: np.ndarray, device: torch.device) -> torch.Tensor:
    """(1,8,T) / (N,1,8,T) 单窗 → (1,8,T) tensor（与 infer.predict_windows 一致）。"""
    arr = np.asarray(w, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        xb = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        xb = arr[0]
    elif arr.ndim == 3:
        xb = arr
    else:
        raise ValueError(f"意外窗 shape={arr.shape}")
    return torch.from_numpy(xb[None, ...]).to(device)


def _forward_logits(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch)
    if logits.ndim > 2:
        logits = logits.reshape(logits.shape[0], -1)
    return logits


@torch.no_grad()
def stream_predict_adabn(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    *,
    update_bn: bool = True,
    version: str = ADABN_VERSION_DEFAULT,
    predict_first: bool = False,
) -> StreamPred:
    """按窗因果流式推理。"""
    freeze_all_params(model)
    param_snap = snapshot_params(model)
    preds: list[int] = []
    probs: list[float] = []
    lat_ms: list[float] = []
    ver = (version or ADABN_VERSION_DEFAULT).lower()

    for i in range(len(X)):
        import time

        t0 = time.perf_counter()
        batch = _window_batch_tensor(X[i], device)

        if ver == "v1.0":
            # 已弃用：单窗 train 归一化
            model.train() if update_bn else model.eval()
            logits = _forward_logits(model, batch)
        elif predict_first or ver == "v1.2":
            # 严格因果：先用旧 running stats 预测，再 train 更新
            model.eval()
            logits = _forward_logits(model, batch)
            if update_bn:
                model.train()
                _ = _forward_logits(model, batch)
        elif ver == "v1.1":
            # v1.1：先更新 running stats，再 eval 预测
            if update_bn:
                model.train()
                _ = _forward_logits(model, batch)
            model.eval()
            logits = _forward_logits(model, batch)
        else:
            raise ValueError(f"未知 AdaBN version: {version!r}")

        prob = torch.softmax(logits, dim=1)[0]
        preds.append(int(prob.argmax().item()))
        probs.append(float(prob.max().item()))
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    assert_params_unchanged(param_snap, snapshot_params(model))
    return StreamPred(
        pred=np.asarray(preds, dtype=np.int64),
        prob_max=np.asarray(probs, dtype=np.float64),
        latency_ms=lat_ms,
    )


def confidence_keep(prob_max: np.ndarray, tau: float | None) -> np.ndarray:
    if tau is None:
        return np.ones(len(prob_max), dtype=bool)
    return np.asarray(prob_max >= float(tau), dtype=bool)
