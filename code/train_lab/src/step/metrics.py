from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_task_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """正类=1(任务)，负类=0(静息)。"""
    # y_true / y_pred 已是类别编号 (N,)，不要用 argmax
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def format_task_metrics(part_name: str, m: dict[str, float]) -> str:
    """part_name: 如 'val' / 'train'，表示打印的是哪一份。"""
    return "\n".join(
        [
            f"===== [{part_name}] 分类头1（静息=0 / 任务=1）braindecode EEGNet =====",
            f"  混淆矩阵: TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}",
            f"  Accuracy      分类准确率   = {m['accuracy']:.4f}",
            f"  Recall        召回率/灵敏度 = {m['recall']:.4f}",
            f"  Specificity   特异性       = {m['specificity']:.4f}",
            f"  Precision     精确率       = {m['precision']:.4f}",
            f"  F1-score      F1          = {m['f1']:.4f}",
            f"  Balanced Acc  平衡准确率   = {m['balanced_accuracy']:.4f}",
            f"  ※ 第二分类头请用 train_three.py / three_class_metrics",
        ]
    )


def three_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """空闲=0 / 左手=1 / 右手=2。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    recall_macro = recall_score(
        y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0
    )
    recall_per = recall_score(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "recall_macro": float(recall_macro),
        "recall_idle": float(recall_per[0]),
        "recall_left": float(recall_per[1]),
        "recall_right": float(recall_per[2]),
        "cm": cm,
    }


def format_three_metrics(part_name: str, m: dict) -> str:
    cm = m["cm"]
    return "\n".join(
        [
            f"===== [{part_name}] 分类头2（空闲=0 / 左=1 / 右=2） =====",
            f"  混淆矩阵 (行=真实, 列=预测):",
            f"            pred0  pred1  pred2",
            f"    true0  {cm[0, 0]:5d}  {cm[0, 1]:5d}  {cm[0, 2]:5d}",
            f"    true1  {cm[1, 0]:5d}  {cm[1, 1]:5d}  {cm[1, 2]:5d}",
            f"    true2  {cm[2, 0]:5d}  {cm[2, 1]:5d}  {cm[2, 2]:5d}",
            f"  Accuracy     = {m['accuracy']:.4f}",
            f"  F1-macro     = {m['f1_macro']:.4f}",
            f"  Recall-macro = {m['recall_macro']:.4f}",
            f"  Recall idle/left/right = "
            f"{m['recall_idle']:.4f} / {m['recall_left']:.4f} / {m['recall_right']:.4f}",
        ]
    )
