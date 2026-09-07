"""方案25 回归：G1 val 无增广、epoch 多样性、trial 平票、pack meta。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "5070_baselines_openbmi_3s_hop100_accpaper"
for p in (str(BASE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from domain_aug import AugConfig, apply_domain_aug_np, aug_config_g1  # noqa: E402
from patch_baseline import patch_baseline_modules, set_aug_config, set_aug_epoch  # noqa: E402


def test_val_dataset_no_augmentation() -> None:
    set_aug_config(
        AugConfig(enabled=True, p_apply=1.0, ops=("noise",), noise_sigmas=(0.5,)),
        seed=7,
    )
    patch_baseline_modules(train_device="5070")
    import task_runner as tr

    td = tempfile.mkdtemp()
    try:
        p = Path(td) / "x.npy"
        x = np.zeros((4, 8, 10), dtype=np.float32)
        np.save(p, x)
        y = np.array([0, 1, 2, 0], dtype=np.int64)
        train_ds = tr.PackedArrayDataset(y, x_path=p, augment=True)
        val_ds = tr.PackedArrayDataset(y, x_path=p, augment=False)
        set_aug_epoch(1)
        xt, _ = train_ds[0]
        xv, _ = val_ds[0]
        del train_ds, val_ds
        assert not np.allclose(xt.numpy(), xv.numpy()), "train should be augmented"
        assert np.allclose(xv.numpy(), 0.0), "val must stay clean"
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def test_epoch_changes_augmentation() -> None:
    cfg = AugConfig(enabled=True, p_apply=1.0, ops=("noise",))
    x = np.ones((8, 16), dtype=np.float32)
    a = apply_domain_aug_np(x, cfg, seed=1, index=3, epoch=0)
    b = apply_domain_aug_np(x, cfg, seed=1, index=3, epoch=1)
    assert not np.allclose(a, b), "不同 epoch 应产生不同噪声"


def test_trial_metrics_idle_tie_not_biased_to_left() -> None:
    from trial_metrics import aggregate_windows_to_trials

    # idle 真标签；窗预测 1/1/0/0 → 1 与 0 平票 → 应取 min=0（非旧 (0+1)%3=1）
    y = np.array([0, 0, 0, 0])
    p = np.array([1, 1, 0, 0])
    r = aggregate_windows_to_trials(
        y, p, np.array(["S"] * 4), np.array([1, 1, 1, 1]), n_classes=3
    )
    assert int(r["trial_pred_majority"][0]) == 0
    assert r["metrics"]["acc_majority"] == 1.0


def test_pack_meta_rejects_stale_indices() -> None:
    import task_runner as tr

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = np.random.randn(20, 1, 8, 12).astype(np.float32)
        idx_a = np.array([1, 2, 3], dtype=np.int64)
        idx_b = np.array([4, 5, 6], dtype=np.int64)
        pack = td_path / "pack.npy"
        tr.materialize_time_pack(src, idx_a, pack)
        meta = json.loads(tr._pack_meta_path(pack).read_text(encoding="utf-8"))
        assert meta["dtype"] == "float32"
        assert meta["indices_hash"] == tr._indices_fingerprint(idx_a, 12, np.float32)["indices_hash"]
        assert not tr._pack_meta_matches(tr._pack_meta_path(pack), idx_b, 12, np.float32)


if __name__ == "__main__":
    test_val_dataset_no_augmentation()
    test_epoch_changes_augmentation()
    test_trial_metrics_idle_tie_not_biased_to_left()
    test_pack_meta_rejects_stale_indices()
    print("test_s25_fixes: OK")
