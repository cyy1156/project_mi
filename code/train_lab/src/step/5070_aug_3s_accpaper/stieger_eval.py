"""Stieger 评测：09 v1.2 noz_unified eval_half（无 EA/AdaBN）。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet

from s25_config import N_FOLDS, N_TIMES  # noqa: E402
from data import SubjectStream  # noqa: E402
from data_split import build_cue_split, window_mask_for_cues  # noqa: E402
from ea import prepare_zscore_eval_batch  # noqa: E402
from infer import load_fold_model, predict_windows  # noqa: E402
from otta_infer import build_eval_tensors, predict_eval_pack  # noqa: E402
from util_metrics import aggregate_windows_to_segments  # noqa: E402


def build_shallow(n_chans, n_times, n_outputs, drop_prob=0.5):
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def eval_subject_fold(
    stream: SubjectStream,
    *,
    init_run,
    head: str,
    fold: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """后半 eval_half · noz → z-score · 无 EA/AdaBN（对齐 09-A0）。"""
    split = build_cue_split(stream, val_ratio=0.2, seed=42)
    pack = build_eval_tensors(stream, split, ea_ref="off", r_ref_src=None)
    n_classes = 2 if head == "task" else 3

    net = load_fold_model(
        build_shallow, init_run, head=head, fold=fold, device=device
    )
    out = predict_eval_pack(
        net,
        pack,
        device,
        adabn=False,
        conf_tau=None,
        batch_size=batch_size,
    )
    pred = out["pred"]
    agg = aggregate_windows_to_segments(
        pack["y_task"] if head == "task" else pack["y_three"],
        pred,
        pack["seg_keys"],
        n_classes=n_classes,
    )
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "fold": fold,
        "acc_paper": float(agg["segment_metrics"]["acc_paper"]),
        "segment_metrics": agg["segment_metrics"],
        "window_metrics": agg["window_metrics"],
        "n_eval_windows": int(pack["n_eval"]),
        "eval_cues": split.eval_cues,
    }


def eval_mask_pack(
    stream: SubjectStream,
    mask_eval: np.ndarray,
    model,
    *,
    head: str,
    device: torch.device,
    batch_size: int = 64,
    use_noz_pipeline: bool = True,
) -> dict:
    n_classes = 2 if head == "task" else 3
    y = stream.y_task if head == "task" else stream.y_three
    eval_idx = np.flatnonzero(mask_eval)
    if len(eval_idx) == 0:
        raise RuntimeError("eval_mask 为空")
    if use_noz_pipeline and stream.X_noz is not None:
        X_eval = prepare_zscore_eval_batch(stream.X_noz, eval_idx)
        pred = predict_windows(model, X_eval, device, batch_size=batch_size)
    else:
        pred = predict_windows(
            model, stream.X[eval_idx], device, batch_size=batch_size
        )
    return aggregate_windows_to_segments(
        y[eval_idx], pred, stream.seg_keys[eval_idx], n_classes=n_classes
    )
