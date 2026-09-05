"""Scheme 16 · 5090 · OpenBMI Acc_paper · shallow + hierarchical Three loss.

对照机 RTX 5090：系统内存 128GB · 显存 32GB；全量五折友好默认（workers + pin_memory）。
正式旁路试探仍以 5060 包为准；本包结果默认登记为对照。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5090"
TRAIN_DEVICE_NOTE = "RAM 128GB · VRAM 32GB · scheme16 hier-loss full chain"
OUT_ROOT_TAG = "5090_three_hier_loss_accpaper"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_2s_hop100"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    batch_train: int = 256
    batch_eval: int = 512
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss "
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train "
        "device=5090"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    openbmi_only: bool = True
    # 128GB 内存：不必长期留 pack；折结束可删（省盘）
    keep_fold_packs: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking: bool = True
    torch_num_threads: int = 8
    cudnn_benchmark: bool = True
    deterministic: bool = False
    use_amp: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
