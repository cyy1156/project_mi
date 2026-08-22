"""OTTA 推理：EA 预处理 + 流式 AdaBN + 置信度掩码。"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from adabn import confidence_keep, restore_bn_running, snapshot_bn_running, stream_predict_adabn
from ea import prepare_ea_eval_batch, spatial_cov_avg
from ref_cov import load_ref_cov_src


def build_eval_tensors(
    stream,
    split,
    *,
    ea_ref: str | None,
    r_ref_src: np.ndarray | None,
) -> dict[str, Any]:
    """构建后半评测窗与元数据。"""
    from data_split import window_mask_for_cues  # noqa: WPS433

    cal_mask = window_mask_for_cues(stream.cue_ids, split.train_cues)
    eval_mask = window_mask_for_cues(stream.cue_ids, split.eval_cues)
    cal_idx = np.flatnonzero(cal_mask)
    eval_idx = np.flatnonzero(eval_mask)
    if len(eval_idx) == 0:
        raise RuntimeError(f"{stream.subject_id}: 后半无窗")

    x_noz = stream.X_noz
    if x_noz is None:
        raise RuntimeError(f"{stream.subject_id}: 缺少 X_noz")

    if ea_ref is None:
        X_eval = stream.X[eval_idx]
    elif ea_ref == "src":
        if r_ref_src is None:
            r_ref_src, _ = load_ref_cov_src()
        X_eval = prepare_ea_eval_batch(
            x_noz, cal_idx, eval_idx, r_ref=r_ref_src
        )
    elif ea_ref == "cal":
        r_cal = spatial_cov_avg(x_noz[cal_idx])
        X_eval = prepare_ea_eval_batch(
            x_noz, cal_idx, eval_idx, r_ref=r_cal, r_src=r_cal
        )
    else:
        raise ValueError(ea_ref)

    return {
        "cal_idx": cal_idx,
        "eval_idx": eval_idx,
        "X_eval": X_eval,
        "X_noz_eval": x_noz[eval_idx],
        "y_task": stream.y_task[eval_idx],
        "y_three": stream.y_three[eval_idx],
        "seg_keys": stream.seg_keys[eval_idx],
        "cue_ids": stream.cue_ids[eval_idx],
        "segs": stream.segs[eval_idx],
        "n_eval": int(len(eval_idx)),
    }


def predict_eval_pack(
    model: nn.Module,
    pack: dict[str, Any],
    device: torch.device,
    *,
    adabn: bool,
    conf_tau: float | None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """对单 fold 模型在 eval 窗上推理。"""
    from infer import predict_windows  # noqa: WPS433

    X = pack["X_eval"]
    bn0 = snapshot_bn_running(model)
    model2 = model  # 原地 AdaBN

    if adabn:
        sp = stream_predict_adabn(
            model2, X, device, update_bn=True, conf_tau=None
        )
        pred = sp.pred
        prob_max = sp.prob_max
        lat = sp.latency_ms
    else:
        pred = predict_windows(model2, X, device, batch_size=batch_size)
        prob_max = np.ones(len(pred), dtype=np.float64)
        lat = []

    conf_k = confidence_keep(prob_max, conf_tau)
    restore_bn_running(model, bn0)

    return {
        "pred": pred,
        "prob_max": prob_max,
        "conf_keep": conf_k,
        "latency_ms": lat,
        "latency_mean_ms": float(np.mean(lat)) if lat else float("nan"),
    }


def clone_model_for_subject(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)
