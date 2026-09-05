"""加载 S3 ckpt / FT ckpt 并对窗推理。"""

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
    model.eval()
    if X.ndim == 4 and X.shape[1] == 1:
        Xb = X[:, 0, :, :]
    elif X.ndim == 3:
        Xb = X
    else:
        raise ValueError(f"意外 X shape={X.shape}")
    preds: list[np.ndarray] = []
    for i in range(0, len(Xb), batch_size):
        batch = torch.from_numpy(
            np.asarray(Xb[i : i + batch_size], dtype=np.float32)
        ).to(device)
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
    ckpt_n = ckpt.get("n_outputs")
    if ckpt_n is not None and int(ckpt_n) != n_out:
        raise RuntimeError(f"{path}: n_outputs={ckpt_n} 与 head={head} 不符")
    early = ckpt.get("early_stop")
    if early is not None and early != "acc_paper":
        raise RuntimeError(f"{path}: early_stop={early!r}，期望 acc_paper")
    n_times = int(hp.get("n_times_expected") or hp.get("n_times") or N_TIMES)
    if n_times != N_TIMES:
        # 允许 ckpt 未写 n_times；硬校验输入窗长
        n_times = N_TIMES
    model = build_model(8, n_times, n_out, drop).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    return model


def load_ft_fold(
    build_model: BuildFn,
    *,
    ft_subject_root: Path,
    head: str,
    fold: int,
    device: torch.device,
) -> nn.Module:
    path = ft_subject_root / head / f"fold{fold}" / f"best_{head}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams") or {}
    drop = float(hp.get("drop_prob", 0.5))
    n_out = 2 if head == "task" else 3
    model = build_model(8, N_TIMES, n_out, drop).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    return model
