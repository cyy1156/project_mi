"""OpenBMI Acc_paper：Val Acc_paper 早停；balbatch；patience=20；全 11 模型（RTX 5090）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

TRAIN_DEVICE_LABEL = "NVIDIA RTX 5090"
TRAIN_DEVICE_NOTE = "RAM 128GB · VRAM 32GB · sm_120 · conda cyy · PyTorch 2.11+cu128"
OUT_ROOT_TAG = "5090_baseline_openbmi_2s_hop100_accpaper"


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
        "2s-hop100ms-balbatch-accpaper-openbmi "
        "subject_key=openbmi:subjNN sess01+02"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    openbmi_only: bool = True
    # GPU / DataLoader（Windows 建议 num_workers=4；双项目同机可改为 2）
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    non_blocking: bool = True
    use_amp: bool = True
    cudnn_benchmark: bool = False
    gpu_memory_fraction: float = 1  # 双任务；独占 GPU 时改为 1.0


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
