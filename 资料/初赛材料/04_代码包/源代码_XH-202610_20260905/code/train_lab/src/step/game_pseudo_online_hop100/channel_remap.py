"""游戏通道序 → OpenBMI 训练序。"""

from __future__ import annotations

import numpy as np

# 与 5060 OpenBMI shallow / channel_fe 一致
OPENBMI_CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]


def remap_windows_to_openbmi(
    X: np.ndarray,
    src_ch_names: list[str],
    *,
    dst_ch_names: list[str] | None = None,
) -> np.ndarray:
    """将窗重排到 OpenBMI 通道序。

    X: (N,1,C,T) 或 (N,C,T)
    """
    dst = list(dst_ch_names or OPENBMI_CHANS)
    src = [str(c) for c in src_ch_names]
    if len(src) != len(dst):
        raise ValueError(f"通道数不符: src={len(src)} dst={len(dst)}")
    try:
        idx = [src.index(c) for c in dst]
    except ValueError as e:
        raise ValueError(f"缺少目标通道: src={src} dst={dst}") from e

    if X.ndim == 4 and X.shape[1] == 1:
        return np.ascontiguousarray(X[:, :, idx, :])
    if X.ndim == 3:
        return np.ascontiguousarray(X[:, idx, :])
    raise ValueError(f"意外 X shape={X.shape}")
