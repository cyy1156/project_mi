"""按 group 划分训练 / 测试集（避免滑窗泄漏）。"""

from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def split_train_test(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_retries: int = 50,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    按 group 留出 test_size 比例作测试集；同一 group 的样本只出现在 train 或 test 一侧。

    返回 train_idx, test_idx, split_info。
    """
    n_groups = len(np.unique(groups))
    if n_groups < 5:
        raise RuntimeError(f"分组数 {n_groups} 过少，无法稳定划分 20% 测试集")

    labels = np.unique(y)
    n_classes = len(labels)

    for attempt in range(max_retries):
        seed = random_state + attempt
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(np.zeros(len(y)), y, groups))

        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < n_classes or len(np.unique(y_test)) < n_classes:
            continue

        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])
        if len(np.intersect1d(train_groups, test_groups)) > 0:
            continue

        info = {
            "test_size": test_size,
            "random_state": seed,
            "n_train_samples": int(len(train_idx)),
            "n_test_samples": int(len(test_idx)),
            "n_train_groups": int(len(train_groups)),
            "n_test_groups": int(len(test_groups)),
            "train_label_counts": dict(Counter(y_train.tolist())),
            "test_label_counts": dict(Counter(y_test.tolist())),
        }
        return train_idx, test_idx, info

    raise RuntimeError(
        f"在 {max_retries} 次尝试内无法得到 train/test 均含全部 {n_classes} 类的划分，"
        "请增加 session 或检查类别样本量。"
    )
