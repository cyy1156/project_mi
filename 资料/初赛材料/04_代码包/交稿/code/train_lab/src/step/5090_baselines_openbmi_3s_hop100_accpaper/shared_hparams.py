"""OpenBMI Acc_paper · 3s/hop100 · 5090（方案 24 · V/T/E 腿）。

超参对齐 5090 方案 21/23：batch 256/512 · AMP · lr/wd 1e-4 · patience 20。
Windows 长链默认 num_workers=0（CLI 覆盖）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5090"
TRAIN_DEVICE_NOTE = (
    "RAM 128GB · VRAM 32GB · scheme24 alg-incr 3s · Tw=3s hop=100ms · batch 256/512"
)
OUT_ROOT_TAG = "5090_alg_incr_3s_hop100_accpaper"
SCHEME24_RUNS_TAG = "5090_openbmi_3s_hop100"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_3s_hop100"
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
        "3s-hop100ms-balbatch-accpaper-openbmi "
        "subject_key=openbmi:subjNN device=5090 scheme24"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 750
    no_rap: bool = True
    no_balbatch: bool = False
    openbmi_only: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    non_blocking: bool = True
    torch_num_threads: int = 8
    cudnn_benchmark: bool = True
    deterministic: bool = False
    use_amp: bool = True
    t0_weight_alpha: float = 0.0
    t0_filter_max: float = 0.0


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
