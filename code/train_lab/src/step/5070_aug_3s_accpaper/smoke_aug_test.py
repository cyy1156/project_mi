"""方案 §1：增广须在 z-score 之后；幅度类增广在 z 前无效。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from domain_aug import AugConfig, apply_domain_aug_np  # noqa: E402

# preprocess 公共 z-score
PRE = HERE.parents[3] / "preprocess_lab" / "src"
if str(PRE) not in sys.path:
    sys.path.insert(0, str(PRE))
from common.steps.resample_zscore import trial_zscore  # noqa: E402


def _make_window() -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(8, 750)).astype(np.float32)
    return trial_zscore(raw)


def test_post_zscore_aug_changes_input() -> None:
    x = _make_window()
    cfg = AugConfig(enabled=True, p_apply=1.0, ops=("noise",))
    y = apply_domain_aug_np(x, cfg, seed=1, index=0)
    assert not np.allclose(x, y), "噪声增广应改变 z-score 后输入"


def test_pre_zscore_amplitude_aug_washed_out() -> None:
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(8, 750)).astype(np.float32)
    scaled = raw * 1.5
    z1 = trial_zscore(raw)
    z2 = trial_zscore(scaled)
    assert np.allclose(z1, z2, atol=1e-5), "z-score 前应抵消幅度缩放"


def test_time_shift_changes_post_zscore() -> None:
    x = _make_window()
    cfg = AugConfig(enabled=True, p_apply=1.0, ops=("time_shift",))
    y = apply_domain_aug_np(x, cfg, seed=3, index=0)
    assert not np.allclose(x, y), "时移增广应改变 z-score 后输入"


if __name__ == "__main__":
    test_post_zscore_aug_changes_input()
    test_pre_zscore_amplitude_aug_washed_out()
    test_time_shift_changes_post_zscore()
    print("smoke_aug_test: OK")
