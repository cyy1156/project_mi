"""5060 · 掩码未来表征预测 + 双专家门控 · OpenBMI Acc_paper（低内存旁路）.

本机 RTX 5060 Laptop / ~16GB RAM：小 batch、workers=0、关 pin。
全量五折 / 大 batch 请用姊妹包：`5090_mask_future_dual_expert_accpaper/`。
方案：资料/模型方案/掩码未来表征预测_双专家门控_在线MI/ 定稿 v1.14+
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"
TRAIN_DEVICE_NOTE = "16GB · scheme17 mask-future · low-mem fold0-first"
OUT_ROOT_TAG = "5060_mask_future_dual_expert_accpaper"

# A0 用旧 500pt；A1+ 用新预处理臂（见数据切片说明）
DATA_TAG_A0 = "openbmi_2s_hop100"
DATA_TAG_PF = "openbmi_2s_hop100_pf1000"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag_a0: str = DATA_TAG_A0
    data_tag_pf: str = DATA_TAG_PF
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    # 16GB / 8GB VRAM：相对方案文档 256/512 降档（定稿允许 OOM 降 batch）
    batch_train: int = 128
    batch_eval: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "mask-future-dual-expert Acc_paper Three-only "
        "Tw=2s hop=100ms postMI>=1.6s subject_key=openbmi:subjNN "
        "Adam device=5060 low-mem"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_chans: int = 8
    n_times_a0: int = 500
    n_times_pf: int = 1000
    n_classes: int = 3  # Three
    embed_dim: int = 40  # D
    lambda_pred: float = 1.0
    lambda_sig: float = 0.05
    lambda_dec: float = 0.2  # P2；P1=0
    lambda_cls: float = 1.0
    sigreg_slices: int = 1024
    pred_dropout: float = 0.3
    openbmi_only: bool = True
    keep_fold_packs: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    non_blocking: bool = False
    torch_num_threads: int = 2
    cudnn_benchmark: bool = False
    deterministic: bool = False
    use_amp: bool = True
    ema_momentum: float = 0.996  # B10


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
