"""试次级聚合：Acc_paper（正确窗占比 >50%）+ 众数 BalAcc/F1。

本文件位于 accpaper 包内，与 trialmaj 复评包解耦（逻辑同构，禁止互相改坏）。
"""

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
    best_n = top[0][1]
    leaders = [c for c, n in cnt.items() if n == best_n]
    if len(leaders) == 1:
        return int(leaders[0])
    # 平票：取最小类 id（确定性、与真标签无关，避免 idle→left 偏置）
    return int(min(leaders))


def aggregate_windows_to_trials(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    *,
    n_classes: int,
) -> dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    subjects = np.asarray([str(s) for s in subjects])
    trial_ids = np.asarray(trial_ids).astype(np.int64).reshape(-1)
    assert len(y_true) == len(y_pred) == len(subjects) == len(trial_ids)

    keys = list(zip(subjects.tolist(), trial_ids.tolist()))
    order: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)

    trial_y: list[int] = []
    trial_pred_maj: list[int] = []
    trial_paper_ok: list[bool] = []
    trial_n_win: list[int] = []
    trial_correct_rate: list[float] = []
    trial_subjects: list[str] = []
    trial_ids_out: list[int] = []

    for subj, tid in order:
        mask = (subjects == subj) & (trial_ids == tid)
        yt = y_true[mask]
        yp = y_pred[mask]
        assert len(yt) > 0
        uniq = np.unique(yt)
        if len(uniq) != 1:
            raise ValueError(
                f"trial ({subj}, {tid}) 标签不恒定: {uniq.tolist()} n={len(yt)}"
            )
        y = int(uniq[0])
        rate = float(np.mean(yp == y))
        paper_ok = rate > 0.5
        maj = _majority_label(yp)
        maj_for_metric = maj if maj != TIE_SENTINEL else int(y)

        trial_y.append(y)
        trial_pred_maj.append(maj_for_metric)
        trial_paper_ok.append(paper_ok)
        trial_n_win.append(int(len(yt)))
        trial_correct_rate.append(rate)
        trial_subjects.append(subj)
        trial_ids_out.append(int(tid))

    yt_arr = np.asarray(trial_y, dtype=np.int64)
    yp_arr = np.asarray(trial_pred_maj, dtype=np.int64)
    paper_ok_arr = np.asarray(trial_paper_ok, dtype=bool)
    acc_paper = float(paper_ok_arr.mean()) if len(paper_ok_arr) else 0.0
    acc_majority = float(np.mean(yp_arr == yt_arr)) if len(yt_arr) else 0.0

    if n_classes == 2:
        m = binary_task_metrics(yt_arr, yp_arr)
        metrics = {
            "n_trials": int(len(yt_arr)),
            "n_windows": int(len(y_true)),
            "acc_paper": acc_paper,
            "acc_majority": acc_majority,
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1": float(m["f1"]),
            "specificity": float(m["specificity"]),
            "recall": float(m["recall"]),
            "precision": float(m["precision"]),
            "accuracy_sklearn": float(m["accuracy"]),
            **{k: m[k] for k in ("tp", "tn", "fp", "fn")},
        }
    elif n_classes == 3:
        m = three_class_metrics(yt_arr, yp_arr)
        metrics = {
            "n_trials": int(len(yt_arr)),
            "n_windows": int(len(y_true)),
            "acc_paper": acc_paper,
            "acc_majority": acc_majority,
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1_macro": float(m["f1_macro"]),
            "precision_macro": float(m["precision_macro"]),
            "recall_macro": float(m["recall_macro"]),
            "accuracy_sklearn": float(m["accuracy"]),
            "recall_idle": float(m["recall_idle"]),
            "recall_left": float(m["recall_left"]),
            "recall_right": float(m["recall_right"]),
            "cm": m["cm"],
        }
    else:
        raise ValueError(f"n_classes={n_classes}")

    return {
        "metrics": jsonify_metrics(metrics),
        "trial_y": yt_arr,
        "trial_pred_majority": yp_arr,
        "trial_paper_ok": paper_ok_arr,
        "trial_n_windows": np.asarray(trial_n_win, dtype=np.int64),
        "trial_correct_rate": np.asarray(trial_correct_rate, dtype=np.float64),
        "trial_subjects": np.asarray(trial_subjects, dtype=object),
        "trial_ids": np.asarray(trial_ids_out, dtype=np.int64),
    }


def unit_check_boundary() -> None:
    y = np.array([1, 1, 1, 1])
    p = np.array([1, 1, 0, 0])
    r = aggregate_windows_to_trials(
        y, p, np.array(["A"] * 4), np.array([0, 0, 0, 0]), n_classes=2
    )
    assert r["metrics"]["acc_paper"] == 0.0
    assert r["metrics"]["acc_majority"] == 0.0

    p2 = np.array([1, 1, 1, 0])
    r2 = aggregate_windows_to_trials(
        y, p2, np.array(["A"] * 4), np.array([0, 0, 0, 0]), n_classes=2
    )
    assert r2["metrics"]["acc_paper"] == 1.0
    assert r2["metrics"]["acc_majority"] == 1.0
    print("accpaper trial_metrics unit_check OK")


if __name__ == "__main__":
    unit_check_boundary()
