"""单元测试：不依赖真实 mat。"""
from __future__ import annotations

import numpy as np

from src.common.steps.resample_zscore import to_model_tensor
from src.datasets.openbmi_pf1000.pipeline import (
    FUT,
    N_TIMES,
    windows_from_seg250,
    sanity_check_pf1000,
)


def test_windows_geometry():
    rng = np.random.default_rng(0)
    seg = rng.normal(size=(1400, 8)).astype(np.float32)
    wins = windows_from_seg250(seg, lab_task=1, lab_three=1, zscore=True)
    # t0=0.4..2.0 hop0.1 → 17
    assert len(wins) == 17, len(wins)
    x, yt, y3, t0 = wins[0]
    assert x.shape == (1000, 8)
    assert abs(t0 - 0.4) < 1e-6
    assert yt == 1 and y3 == 1
    # 缺 past 的 i0=0 不应出现
    assert all(w[3] >= 0.4 - 1e-6 for w in wins)


def test_mask_tail_zero():
    rng = np.random.default_rng(1)
    seg = rng.normal(size=(1400, 8)).astype(np.float32)
    wins = windows_from_seg250(seg, lab_task=1, lab_three=2, zscore=False)
    xs = [w[0] for w in wins]
    X_full = to_model_tensor(xs)
    X_mask = X_full.copy()
    X_mask[..., -FUT:] = 0
    sanity_check_pf1000(X_full, X_mask, np.array([2] * len(xs), dtype=np.int64))
    assert X_full.shape[-1] == N_TIMES
    assert X_mask.shape[-1] == N_TIMES


def test_short_seg_no_windows():
    seg = np.zeros((1000, 8), np.float32)  # <1400
    wins = windows_from_seg250(seg, lab_task=1, lab_three=1, zscore=False)
    assert wins == []


if __name__ == "__main__":
    test_windows_geometry()
    test_mask_tail_zero()
    test_short_seg_no_windows()
    print("OK")
