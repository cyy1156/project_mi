"""段级 Acc_paper / BalAcc_maj + 窗级指标。"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent.parent
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

from metrics import binary_task_metrics, jsonify_metrics, three_class_metrics

TIE_SENTINEL = -1


def _majority_label(preds: np.ndarray) -> int:
    cnt = Counter(int(p) for p in preds.tolist())
    top = cnt.most_common()
    if not top:
        return TIE_SENTINEL
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return TIE_SENTINEL
    return int(top[0][0])


def aggregate_windows_to_segments(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seg_keys: np.ndarray,
    *,
    n_classes: int,
) -> dict:
    """聚合键 = seg_key（如 `12:mi`），等价方案中的段。"""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    seg_keys = np.asarray([str(s) for s in seg_keys])
    assert len(y_true) == len(y_pred) == len(seg_keys)

    order: list[str] = []
    seen: set[str] = set()
    for k in seg_keys.tolist():
        if k not in seen:
            seen.add(k)
            order.append(k)

    trial_y: list[int] = []
    trial_pred_maj: list[int] = []
    trial_paper_ok: list[bool] = []

    for key in order:
        mask = seg_keys == key
        yt = y_true[mask]
        yp = y_pred[mask]
        uniq = np.unique(yt)
        if len(uniq) != 1:
            raise ValueError(f"segment {key} 标签不恒定: {uniq.tolist()}")
        y = int(uniq[0])
        rate = float(np.mean(yp == y))
        paper_ok = rate > 0.5
        maj = _majority_label(yp)
        if maj == TIE_SENTINEL:
            maj_for_metric = (y + 1) % n_classes
        else:
            maj_for_metric = maj
        trial_y.append(y)
        trial_pred_maj.append(maj_for_metric)
        trial_paper_ok.append(paper_ok)

    yt_arr = np.asarray(trial_y, dtype=np.int64)
    yp_arr = np.asarray(trial_pred_maj, dtype=np.int64)
    paper_ok_arr = np.asarray(trial_paper_ok, dtype=bool)
    acc_paper = float(paper_ok_arr.mean()) if len(paper_ok_arr) else 0.0

    if n_classes == 2:
        m = binary_task_metrics(yt_arr, yp_arr)
        win_m = binary_task_metrics(y_true, y_pred)
        seg_metrics = {
            "n_segments": int(len(yt_arr)),
            "n_windows": int(len(y_true)),
            "acc_paper": acc_paper,
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1": float(m["f1"]),
            "specificity": float(m["specificity"]),
            "recall": float(m["recall"]),
            "precision": float(m["precision"]),
            "accuracy": float(m["accuracy"]),
        }
        window_metrics = jsonify_metrics(win_m)
    elif n_classes == 3:
        m = three_class_metrics(yt_arr, yp_arr)
        win_m = three_class_metrics(y_true, y_pred)
        seg_metrics = {
            "n_segments": int(len(yt_arr)),
            "n_windows": int(len(y_true)),
            "acc_paper": acc_paper,
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1_macro": float(m["f1_macro"]),
            "recall_idle": float(m["recall_idle"]),
            "recall_left": float(m["recall_left"]),
            "recall_right": float(m["recall_right"]),
            "accuracy": float(m["accuracy"]),
        }
        window_metrics = jsonify_metrics(win_m)
    else:
        raise ValueError(n_classes)

    return {
        "segment_metrics": jsonify_metrics(seg_metrics),
        "window_metrics": window_metrics,
    }


def mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())
