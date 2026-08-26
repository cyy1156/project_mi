"""方案 26 · 冒烟：E1 融合核 + 特征维 + patch_recipe。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG24 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
for p in (HERE, PKG24):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from e1_fusion_core import fuse_pipeline, calibrate_members  # noqa: E402
from patch_recipe import install_recipe  # noqa: E402
from s26_hparams import RecipeTrainHP  # noqa: E402


def _fake_member(n: int = 200) -> dict:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=n)
    probs = rng.dirichlet([1, 1, 1], size=n).astype(np.float32)
    return {
        "subject": np.array([f"s{i % 5}" for i in range(n)], dtype=object),
        "fold": np.zeros(n, dtype=np.int64),
        "split": np.array(["val"] * (n // 2) + ["test"] * (n - n // 2)),
        "trial_id": np.arange(n) // 11,
        "t0_sec": (np.arange(n) % 11).astype(np.float32) * 0.1,
        "y": y.astype(np.int64),
        "pred": probs.argmax(axis=1).astype(np.int64),
        "p_max": probs.max(axis=1).astype(np.float32),
        "probs": probs,
    }


def test_e1_fusion_smoke() -> None:
    members = [_fake_member(), _fake_member(), _fake_member()]
    temps = calibrate_members(members)
    fused = fuse_pipeline(members, temperatures=temps, weights=(1 / 3, 1 / 3, 1 / 3), smooth_radius=1)
    assert fused["probs"].shape[1] == 3


def test_recipe_patch() -> None:
    hp = install_recipe("R1")
    assert isinstance(hp, RecipeTrainHP)
    assert hp.optimizer == "adamw"
    hp2 = install_recipe("R2")
    assert hp2.use_swa is True


if __name__ == "__main__":
    test_e1_fusion_smoke()
    test_recipe_patch()
    print("smoke_s26_test: OK")
