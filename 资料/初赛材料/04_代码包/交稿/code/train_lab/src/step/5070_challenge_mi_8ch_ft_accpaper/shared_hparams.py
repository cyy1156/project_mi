"""Exp34 轨 B · 官方 8ch · OpenBMI 热启动 / scratch · 5070。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


TRAIN_DEVICE_LABEL = "NVIDIA GeForce RTX 5070 Laptop"
OUT_ROOT_TAG = "5070_challenge_mi_8ch_ft_accpaper"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "challenge_mi_3s_8ch"
    n_folds: int = 6
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    batch_train: int = 128
    batch_eval: int = 256
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = "challenge_mi_3s_8ch LOSO6 device=5070 batch128 OpenBMI-ft"
    train_sampler: str = "balanced_invfreq"
    n_chans_expected: int = 8
    n_times_expected: int = 750
    n_outputs: int = 3
    num_workers: int = 0
    pin_memory: bool = False
    use_amp: bool = True
    torch_num_threads: int = 8
    cudnn_benchmark: bool = False
    non_blocking: bool = True
    init_from_openbmi: bool = True
    remap_openbmi_labels: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)


def hp_scratch(hp: SharedTrainHP | None = None) -> SharedTrainHP:
    base = hp or SHARED
    return SharedTrainHP(
        **{
            **asdict(base),
            "init_from_openbmi": False,
            "protocol": base.protocol.replace("OpenBMI-ft", "scratch"),
        }
    )


def hp_conformer(hp: SharedTrainHP | None = None) -> SharedTrainHP:
    base = hp or SHARED
    return SharedTrainHP(
        **{
            **asdict(base),
            "batch_train": 96,
            "batch_eval": 192,
            "protocol": base.protocol + " conformer96",
        }
    )


def hp_with_seed(seed: int, hp: SharedTrainHP | None = None) -> SharedTrainHP:
    base = hp or SHARED
    return SharedTrainHP(**{**asdict(base), "seed": int(seed)})
