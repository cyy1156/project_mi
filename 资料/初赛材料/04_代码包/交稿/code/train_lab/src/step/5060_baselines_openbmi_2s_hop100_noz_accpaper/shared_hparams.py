"""OpenBMI Acc_paper · 无 z-score 滑窗（方案 07）。对齐正式 5060 HP。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _default_num_workers() -> int:
    # 16GB 本机：4 workers + persistent + 折内 float16 pack 易触发 Windows 1455
    # （Couldn't open shared file mapping）。训练仍可 CLI --num-workers 覆盖。
    return 2


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"
TRAIN_DEVICE_NOTE = "8GB · ~16GB RAM · AMP / cudnn.benchmark defaults"
OUT_ROOT_TAG = "5060_baseline_openbmi_2s_hop100_noz_accpaper"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_2s_hop100_noz"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    batch_train: int = 128
    batch_eval: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "2s-hop100ms-nozscore-balbatch-accpaper-openbmi "
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    openbmi_only: bool = True
    num_workers: int = _default_num_workers()
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking: bool = True
    torch_num_threads: int = 6
    cudnn_benchmark: bool = True
    deterministic: bool = False
    use_amp: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
