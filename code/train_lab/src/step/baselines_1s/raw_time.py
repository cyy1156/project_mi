"""1 s 原始时域整理：(N,1,8,250) → (N,8,250)，不做 bandpower。"""

from __future__ import annotations

import numpy as np

N_TIMES_1S = 250


def squeeze_raw_1s(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    assert X.ndim == 3 and X.shape[1] == 8, X.shape
    assert X.shape[2] == N_TIMES_1S, f"期望 n_times={N_TIMES_1S}, got {X.shape}"
    return X
