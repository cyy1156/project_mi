"""方案 19 · 双频带 Shallow + Gate · HP 对齐方案 18 shared_hparams。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _default_num_workers() -> int:
    return 2


TRAIN_DEVICE_LABEL = "NVIDIA GeForce RTX 5070 Laptop"
TRAIN_DEVICE_NOTE = "8GB VRAM · scheme19 · batch 256/512 · workers=2 · AMP"
OUT_ROOT_TAG = "5070_dual_band_shallow_accpaper"
SCHEME19_RUNS_TAG = "5070_dual_band_shallow"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag_mu: str = "openbmi_2s_hop100_mu813"
    data_tag_beta: str = "openbmi_2s_hop100_beta1330"
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
        "2s-hop100ms-balbatch-accpaper-openbmi-dual-band-shallow "
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070"
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
    # 方案19
    n_filters_branch: int = 20
    lambda_aux: float = 0.5
    fuse: str = "gate"  # gate | fixed05 | concat


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
