"""轻量指标：三分类 Acc / macro-R / macro-S（特异性）/ 混淆矩阵。"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    yt = np.asarray(y_true, dtype=int).reshape(-1)
    yp = np.asarray(y_pred, dtype=int).reshape(-1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(yt.tolist(), yp.tolist()):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def three_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = np.asarray(y_true, dtype=int).reshape(-1)
    yp = np.asarray(y_pred, dtype=int).reshape(-1)
    n = int(len(yt))
    cm = confusion_matrix(yt, yp, 3)
    acc = float((yt == yp).mean()) if n else 0.0

    recalls: list[float] = []
    specs: list[float] = []
    for c in range(3):
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recalls.append(rec)
        specs.append(spe)

    return {
        "n": n,
        "acc": acc,
        "acc_paper": acc,  # 单窗协议下与 Acc 相同
        "macro_recall": float(np.mean(recalls)),
        "macro_specificity": float(np.mean(specs)),
        "per_class_recall": recalls,
        "per_class_specificity": specs,
        "confusion_matrix": cm.tolist(),
    }
