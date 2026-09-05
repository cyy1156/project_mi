"""Task 训练集 SMOTE：仅用于 train 折；val/test 禁止调用。

对 (N,8,T) 展平为 (N,8*T) 做 SMOTE，再还原形状。
依赖：imbalanced-learn。
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE


def smote_resample_eeg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """对二分类 EEG 训练集做 SMOTE，使少数类数量对齐多数类。

    Parameters
    ----------
    X : (N, 8, T) 或 (N, 1, 8, T)
    y : (N,) 标签 0/1

    Returns
    -------
    X_res : (N', 8, T) float32
    y_res : (N',) int64
    info : 采样前后计数等元信息
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int).reshape(-1)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    assert X.ndim == 3 and X.shape[1] == 8, X.shape
    assert len(X) == len(y)

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    n_min = min(n0, n1)
    k = int(min(k_neighbors, max(n_min - 1, 1)))
    if n_min < 2:
        raise ValueError(f"SMOTE 需要每类至少 2 个样本，当前 n0={n0} n1={n1}")

    n_ch, n_t = int(X.shape[1]), int(X.shape[2])
    X_flat = X.reshape(len(X), n_ch * n_t)
    sm = SMOTE(random_state=random_state, k_neighbors=k, sampling_strategy="auto")
    X_res_flat, y_res = sm.fit_resample(X_flat, y)
    X_res = np.asarray(X_res_flat, dtype=np.float32).reshape(-1, n_ch, n_t)
    y_res = np.asarray(y_res, dtype=np.int64)

    info = {
        "k_neighbors": k,
        "n_before": int(len(y)),
        "n_after": int(len(y_res)),
        "n0_before": n0,
        "n1_before": n1,
        "n0_after": int((y_res == 0).sum()),
        "n1_after": int((y_res == 1).sum()),
    }
    return X_res, y_res, info
