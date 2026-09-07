"""加载 Acc_paper ckpt 并对窗 batch 推理。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import N_TIMES
from weights import ckpt_path

BuildFn = Callable[..., nn.Module]


@torch.no_grad()
def predict_windows(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """X: (N,1,8,500) → pred (N,)"""
    model.eval()
    if X.ndim == 4 and X.shape[1] == 1:
        Xb = X[:, 0, :, :]
    elif X.ndim == 3:
        Xb = X
    else:
        raise ValueError(f"意外 X shape={X.shape}")
    preds: list[np.ndarray] = []
    for i in range(0, len(Xb), batch_size):
        batch = torch.from_numpy(np.asarray(Xb[i : i + batch_size], dtype=np.float32)).to(
            device
        )
        logits = model(batch)
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    if not preds:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(preds, axis=0)


def load_fold_model(
    build_model: BuildFn,
    run_dir: Path,
    *,
    head: str,
    fold: int,
    device: torch.device,
) -> nn.Module:
    path = ckpt_path(run_dir, head=head, fold=fold)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams") or {}
    drop = float(hp.get("drop_prob", 0.5))
    n_out = 2 if head == "task" else 3
    # 校验 ckpt 头
    ckpt_n = ckpt.get("n_outputs")
    if ckpt_n is not None and int(ckpt_n) != n_out:
        raise RuntimeError(f"{path}: n_outputs={ckpt_n} 与 head={head} 不符")
    early = ckpt.get("early_stop")
    if early is not None and early != "acc_paper":
        raise RuntimeError(f"{path}: early_stop={early!r}，期望 acc_paper")
    model = build_model(8, N_TIMES, n_out, drop).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model
