"""在调用 baseline task_runner 前注入增广与 out 前缀。"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from domain_aug import AugConfig, apply_domain_aug_np
from s25_config import aug_out_root_tag

if TYPE_CHECKING:
    pass

_AUG_CFG: AugConfig | None = None
_AUG_SEED: int = 42
_AUG_EPOCH: int = 0
_PATCHED = False


def set_aug_config(cfg: AugConfig | None, *, seed: int = 42) -> None:
    global _AUG_CFG, _AUG_SEED
    _AUG_CFG = cfg
    _AUG_SEED = int(seed)


def set_aug_epoch(epoch: int) -> None:
    global _AUG_EPOCH
    _AUG_EPOCH = int(epoch)


def patch_baseline_modules(
    *,
    train_device: str = "5070",
    fast_batch: bool = False,
) -> str:
    """幂等：改 baseline OUT_ROOT_TAG 与 PackedArrayDataset.__getitem__。

    返回实际 out 前缀（5070_aug / 5090_aug）。
    """
    global _PATCHED
    out_tag = aug_out_root_tag(train_device)
    if _PATCHED:
        return out_tag
    import shared_hparams as sh  # baseline 包内（5070_baselines）

    sh.OUT_ROOT_TAG = out_tag
    dev = (train_device or "5070").strip().lower()
    proto_suffix = " scheme25_aug"
    if dev in ("5090", "90"):
        proto_suffix += " train_device=5090"
    hp = sh.SHARED
    if dev in ("5090", "90"):
        # 默认同 5070 S3 HP（可比）；--fast-batch 才加大 batch 提速
        if fast_batch:
            hp = replace(
                hp,
                batch_train=512,
                batch_eval=1024,
                num_workers=4,
                protocol=hp.protocol + proto_suffix,
            )
        else:
            hp = replace(
                hp,
                num_workers=4,
                protocol=hp.protocol + proto_suffix,
            )
    else:
        hp = replace(hp, protocol=hp.protocol + proto_suffix)
    sh.SHARED = hp

    import task_runner as tr

    _orig_init = tr.PackedArrayDataset.__init__
    _orig_getitem = tr.PackedArrayDataset.__getitem__
    _orig_train_one_fold = tr.train_one_fold

    def _init(self, y_pack, *, x_path=None, augment: bool = False, **kwargs):
        _orig_init(self, y_pack, x_path=x_path, augment=augment, **kwargs)
        # 仅 train 显式 augment=True；val/test 不得回落全局增广
        self._aug_cfg = _AUG_CFG if augment and _AUG_CFG is not None else None

    def _getitem(self, i: int):
        x, y = _orig_getitem(self, i)
        cfg = getattr(self, "_aug_cfg", None)
        if cfg is None or not cfg.enabled:
            return x, y
        if isinstance(x, torch.Tensor):
            arr = x.numpy()
        else:
            arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        x_pool = y_pool = None
        if "mixup" in cfg.ops:
            Xv = self._X_view()
            x_pool = Xv
            y_pool = self.y
        out = apply_domain_aug_np(
            arr,
            cfg,
            seed=_AUG_SEED,
            index=int(i),
            y=int(y.item()) if isinstance(y, torch.Tensor) else int(y),
            epoch=_AUG_EPOCH,
            x_pool=x_pool,
            y_pool=y_pool,
        )
        return torch.from_numpy(out), y

    def _train_one_fold(*args, **kwargs):
        tr.AUG_EPOCH_HOOK = set_aug_epoch
        try:
            return _orig_train_one_fold(*args, **kwargs)
        finally:
            tr.AUG_EPOCH_HOOK = None

    tr.PackedArrayDataset.__init__ = _init  # type: ignore[method-assign]
    tr.PackedArrayDataset.__getitem__ = _getitem  # type: ignore[method-assign]
    tr.train_one_fold = _train_one_fold  # type: ignore[method-assign]
    _PATCHED = True
    return out_tag
