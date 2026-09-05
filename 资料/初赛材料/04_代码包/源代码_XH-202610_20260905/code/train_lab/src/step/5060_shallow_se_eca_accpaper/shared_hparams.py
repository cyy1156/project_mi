"""方案 14 · Shallow 前置 SE/ECA：协议同 Acc_paper；独立 out。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"
TRAIN_DEVICE_NOTE = "8GB · shallow front SE/ECA bypass · AMP / cudnn.benchmark"
OUT_ROOT_TAG = "5060_shallow_se_eca_accpaper"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_2s_hop100"
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
        "2s-hop100ms-balbatch-accpaper-openbmi-shallow-se-eca "
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    openbmi_only: bool = True
    # Windows：动态加载 official runner 时 worker 无法 pickle；锁 0
    num_workers: int = 0
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
