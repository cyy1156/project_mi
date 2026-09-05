"""5090 · 方案 23 机制验证 · OpenBMI Acc_paper.

超参与方案 21（5090_mask_future_dual_expert_accpaper / shared_hparams.py）对齐：
  batch 256/512 · lr/wd 1e-4 · patience 20 · seed 42 · D=40 · drop 0.5
  λ_pred=1.0 · λ_sig=0.05 · sigreg_slices=1024 · pred_dropout=0.3 · AMP
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5090"
TRAIN_DEVICE_NOTE = "RAM 128GB · VRAM 32GB · scheme23 mech-verify · 5090 primary"
OUT_ROOT_TAG = "5090_mech_verify_accpaper"

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
    batch_train: int = 256
    batch_eval: int = 512
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "scheme23-mech-verify Acc_paper Three-only pf1000 v3 "
        "subject_key=openbmi:subjNN Adam device=5090 batch=256/512"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_chans: int = 8
    n_times_pf: int = 1000
    n_classes: int = 3
    embed_dim: int = 40
    lambda_pred: float = 1.0
    lambda_sig: float = 0.05
    lambda_dec: float = 0.2
    lambda_cls: float = 1.0
    sigreg_slices: int = 1024
    pred_dropout: float = 0.3
    openbmi_only: bool = True
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
