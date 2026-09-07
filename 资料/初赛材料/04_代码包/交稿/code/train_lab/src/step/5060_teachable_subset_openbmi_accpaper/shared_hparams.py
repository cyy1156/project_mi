"""06 旁路：可教试次 · 子集评估 / 微调；协议对齐正式 Acc_paper。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _default_num_workers() -> int:
    # Windows + 大 mmap：0 更稳（避免 DataLoader spawn 二次加载 torch DLL 失败）
    return 0


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"
OUT_ROOT_TAG = "5060_teachable_subset_openbmi_accpaper"
SCHEMA_VERSION = "teachable_v1"

# 正式权重 run（只读）
FORMAL_SHALLOW_RUN = (
    "shallow_openbmi_2s_hop100_balbatch_accpaper/openbmi_2s_hop100/run_20260807_135828"
)
FORMAL_EEGNET_RUN = (
    "eegnet_openbmi_2s_hop100_balbatch_accpaper/openbmi_2s_hop100/run_20260806_172218"
)


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_2s_hop100"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    batch_eval: int = 256
    batch_train: int = 128
    lr: float = 1e-4
    ft_lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    max_epochs_ft: int = 50
    patience_ft: int = 10
    n_times_expected: int = 500
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
