"""Shallow 网络结构增强旁路（方案 09）：协议同 Acc_paper；独立 out。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _default_num_workers() -> int:
    return 2  # Windows + float16 pack 下 4 worker 易撑爆 RAM（曾 OpenBLAS OOM）


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"
TRAIN_DEVICE_NOTE = "8GB · shallow net-enhance Three bypass · AMP / cudnn.benchmark"
OUT_ROOT_TAG = "5060_shallow_net_enhance_three_accpaper"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_2s_hop100"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    batch_train: int = 128  # 与 S0 对照锁死；勿为吞吐擅自改
    batch_eval: int = 1024  # 仅影响 val/test 吞吐，不改优化轨迹
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three "
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
    prefetch_factor: int = 4  # 喂饱 GPU；不影响指标
    non_blocking: bool = True
    torch_num_threads: int = 6
    cudnn_benchmark: bool = True
    deterministic: bool = False
    use_amp: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
