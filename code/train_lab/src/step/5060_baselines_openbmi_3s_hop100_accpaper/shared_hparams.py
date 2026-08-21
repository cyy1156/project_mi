"""OpenBMI Acc_paper · 3s/hop100 · 仅 Shallow（实验 20）。

相对正式 2s 基线包：只改窗长 2s→3s（n_times=750）；其余 HP/协议冻结。
正式 2s 结果只读对照，禁止写入 2s out。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _default_num_workers() -> int:
    # 与 2s 包一致：Windows 上 2 worker 最稳；内存紧可 --num-workers 0
    return 2


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"
TRAIN_DEVICE_NOTE = "8GB · experiment 20 · Tw=3s hop=100ms Acc_paper shallow-only"
OUT_ROOT_TAG = "5060_baseline_openbmi_3s_hop100_accpaper"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_3s_hop100"
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
        "3s-hop100ms-balbatch-accpaper-openbmi "
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 750
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
