"""P / L hyperparams — do not mix.

L 轨 SharedTrainHP 字段与 hop100 Acc_paper runner 对齐；OUT 独立旁路。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class P_HP:
    """Paper config; only allowed deviation: n_chans=8."""

    n_chans: int = 8
    n_classes: int = 4
    n_times: int = 1125
    fs: float = 250.0
    win_start_sec: float = 0.5
    win_end_sec: float = 5.0
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 1000
    early_stop_patience: int = 300
    reduce_lr_patience: int = 50
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    n_repeats: int = 10
    dropout: float = 0.3
    weight_decay: float = 0.0
    conv_l2: float = 1e-4
    seed_base: int = 20260813


@dataclass(frozen=True)
class SharedTrainHP:
    """L 轨 = 本室 Acc_paper 协议（与正式 hop100 旁路同形）。"""

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
        "2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L "
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    openbmi_only: bool = True
    num_workers: int = 0  # Windows：动态加载 official runner 时 worker 无法 pickle；L 锁 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking: bool = True
    torch_num_threads: int = 6
    cudnn_benchmark: bool = True
    deterministic: bool = False
    use_amp: bool = True


# aliases
L_HP = SharedTrainHP
P = P_HP()
L = SharedTrainHP()
SHARED = L
OUT_ROOT_TAG = "5060_ciacnet_mi_accpaper"


def p_dict() -> dict:
    return asdict(P)


def l_dict() -> dict:
    return asdict(L)


def shared_as_dict() -> dict:
    return asdict(SHARED)
