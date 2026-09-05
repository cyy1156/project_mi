"""Exp34 轨 A · 官方集 59ch · RTX 5070 Laptop（≈8GB）。

对齐方案 v0.4：6-fold LOSO · batch 32/64 · grad_accum=4 · AMP。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


TRAIN_DEVICE_LABEL = "NVIDIA GeForce RTX 5070 Laptop"
TRAIN_DEVICE_NOTE = (
    "VRAM≈8GB · Exp34 trackA 59ch · LOSO6 · batch32/64 accum4 · AMP · device=5070"
)
OUT_ROOT_TAG = "5070_challenge_mi_59ch_accpaper"
SCHEME_RUNS_TAG = "5070_challenge_mi_59ch"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "challenge_mi_3s_59ch"
    n_folds: int = 6  # LOSO：6 人各做一次 Val
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    batch_train: int = 32
    batch_eval: int = 64
    grad_accum: int = 4  # 有效 batch ≈ 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "challenge_mi_3s_59ch cut-then-filter LOSO6 "
        "device=5070 batch32x4 amp"
    )
    early_stop: str = "acc"  # 单窗=试次，Acc≡Acc_paper
    train_sampler: str = "balanced_invfreq"
    n_chans_expected: int = 59
    n_times_expected: int = 750
    n_outputs: int = 3  # Left/Right/Rest
    num_workers: int = 0
    pin_memory: bool = False
    use_amp: bool = True
    torch_num_threads: int = 8
    cudnn_benchmark: bool = False
    non_blocking: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)


def hp_for_conformer(hp: SharedTrainHP | None = None) -> SharedTrainHP:
    """Conformer 更吃显存：16/32 · accum8。"""
    base = hp or SHARED
    return SharedTrainHP(
        **{
            **asdict(base),
            "batch_train": 16,
            "batch_eval": 32,
            "grad_accum": 8,
            "protocol": base.protocol + " conformer16x8",
        }
    )


def hp_with_seed(seed: int, hp: SharedTrainHP | None = None) -> SharedTrainHP:
    base = hp or SHARED
    return SharedTrainHP(**{**asdict(base), "seed": int(seed)})
