"""段级 Acc_paper + 门控：空窗段 abstain。"""

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


def aggregate_windows_to_segments_gated(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seg_keys: np.ndarray,
    keep: np.ndarray,
    *,
    n_classes: int,
) -> dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    seg_keys = np.asarray([str(s) for s in seg_keys])
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    assert len(y_true) == len(y_pred) == len(seg_keys) == len(keep)

    order: list[str] = []
    seen: set[str] = set()
    for k in seg_keys.tolist():
        if k not in seen:
            seen.add(k)
            order.append(k)

    trial_y: list[int] = []
    trial_pred_maj: list[int] = []
    trial_paper_ok: list[bool] = []
    trial_abstain: list[bool] = []

    for key in order:
        mask = seg_keys == key
        yt_all = y_true[mask]
        uniq = np.unique(yt_all)
        if len(uniq) != 1:
            raise ValueError(f"segment {key} 标签不恒定: {uniq.tolist()}")
        y = int(uniq[0])
        kmask = keep[mask]
        trial_y.append(y)
        if int(kmask.sum()) == 0:
            trial_abstain.append(True)
            trial_paper_ok.append(False)
            trial_pred_maj.append(TIE_SENTINEL)
            continue
        yp = y_pred[mask][kmask]
        rate = float(np.mean(yp == y))
        paper_ok = rate > 0.5
        maj = _majority_label(yp)
        if maj == TIE_SENTINEL:
            maj_for_metric = (y + 1) % n_classes
        else:
            maj_for_metric = maj
        trial_abstain.append(False)
        trial_paper_ok.append(paper_ok)
        trial_pred_maj.append(maj_for_metric)

    yt_arr = np.asarray(trial_y, dtype=np.int64)
    abstain_arr = np.asarray(trial_abstain, dtype=bool)
    paper_ok_arr = np.asarray(trial_paper_ok, dtype=bool)
    n_all = int(len(yt_arr))
    n_abstain = int(abstain_arr.sum())
    n_scored = n_all - n_abstain
    scored_ok = paper_ok_arr[~abstain_arr]
    acc_paper = float(scored_ok.mean()) if n_scored else float("nan")
    n_correct = int(scored_ok.sum()) if n_scored else 0
    abstain_as_wrong_acc = float(n_correct / n_all) if n_all else float("nan")
    abstain_rate = float(n_abstain / n_all) if n_all else 0.0

    scored_yt = yt_arr[~abstain_arr]
    scored_yp = np.asarray(trial_pred_maj, dtype=np.int64)[~abstain_arr]

    if n_scored == 0:
        seg_metrics = {
            "n_segments_all": n_all,
            "n_segments_scored": 0,
            "n_segments_abstain": n_abstain,
            "n_windows": int(len(y_true)),
            "n_windows_kept": int(keep.sum()),
            "acc_paper": float("nan"),
            "abstain_rate": abstain_rate,
            "abstain_as_wrong_acc": abstain_as_wrong_acc,
            "balanced_accuracy": float("nan"),
        }
        window_metrics = {}
    elif n_classes == 2:
        m = binary_task_metrics(scored_yt, scored_yp)
        # 窗级：仅 keep 窗
        w_yt = y_true[keep]
        w_yp = y_pred[keep]
        win_m = binary_task_metrics(w_yt, w_yp) if len(w_yt) else {}
        seg_metrics = {
            "n_segments_all": n_all,
            "n_segments_scored": n_scored,
            "n_segments_abstain": n_abstain,
            "n_windows": int(len(y_true)),
            "n_windows_kept": int(keep.sum()),
            "acc_paper": acc_paper,
            "abstain_rate": abstain_rate,
            "abstain_as_wrong_acc": abstain_as_wrong_acc,
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1": float(m["f1"]),
            "recall": float(m["recall"]),
            "specificity": float(m["specificity"]),
            "accuracy": float(m["accuracy"]),
        }
        window_metrics = jsonify_metrics(win_m) if win_m else {}
    else:
        m = three_class_metrics(scored_yt, scored_yp)
        w_yt = y_true[keep]
        w_yp = y_pred[keep]
        win_m = three_class_metrics(w_yt, w_yp) if len(w_yt) else {}
        seg_metrics = {
            "n_segments_all": n_all,
            "n_segments_scored": n_scored,
            "n_segments_abstain": n_abstain,
            "n_windows": int(len(y_true)),
            "n_windows_kept": int(keep.sum()),
            "acc_paper": acc_paper,
            "abstain_rate": abstain_rate,
            "abstain_as_wrong_acc": abstain_as_wrong_acc,
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1_macro": float(m["f1_macro"]),
            "recall_idle": float(m["recall_idle"]),
            "recall_left": float(m["recall_left"]),
            "recall_right": float(m["recall_right"]),
            "accuracy": float(m["accuracy"]),
            "cm": m["cm"],
        }
        window_metrics = jsonify_metrics(win_m) if win_m else {}

    return {
        "segment_metrics": jsonify_metrics(seg_metrics),
        "window_metrics": window_metrics,
    }
