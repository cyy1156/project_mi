"""OpenBMI · LeJEPA Three 探测旁路 HP（非正式表）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


OUT_ROOT_TAG = "5060_lejepa_three_probe_openbmi_accpaper"
TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"


@dataclass(frozen=True)
class SharedHP:
    data_tag: str = "openbmi_2s_hop100"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    n_times_expected: int = 500
    n_chans: int = 8
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    patch_time: int = 25
    mask_ratio: float = 0.25  # 方案 B：四块并集总遮盖率
    mask_n_blocks: int = 4
    lambda_sigreg: float = 0.05  # 论文单 trade-off；冒烟可扫 {0.02,0.05,0.1}
    num_slices: int = 256
    pretrain_epochs: int = 50
    pretrain_batch: int = 256  # 性能档：吃满 5060；对照 10 号时登记注明
    pretrain_lr: float = 1e-4
    max_epochs: int = 300
    patience: int = 20
    batch_train: int = 256
    batch_eval: int = 512
    lr: float = 1e-4
    lr_encoder: float = 1e-5
    weight_decay: float = 1e-4
    drop_prob: float = 0.5
    num_workers: int = 2  # Windows 稳妥上限；0 太饿 GPU
    pin_memory: bool = True
    use_amp: bool = True


SHARED = SharedHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
