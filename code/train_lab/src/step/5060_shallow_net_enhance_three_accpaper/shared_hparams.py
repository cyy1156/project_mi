"""Shallow 网络结构增强 · 主攻 Three（OpenBMI · Acc_paper）。

旁路 09：协议与正式 5060 完全一致（openbmi_2s_hop100 / z-score / balbatch / Acc_paper），
仅对 ShallowFBCSPNet 做网络结构 / 读出头 / 训练目标消融。
  - 正式出数 = Fast 默认（本机 RTX 5060）
  - 代码 / out / runs 目录独立，不修改正式表
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _default_num_workers() -> int:
    # 折内 float16 pack ~2GB + 源 mmap + pin_memory：Windows 上 2 worker 最稳
    # 内存充裕可 --num-workers 4
    return 2


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"
TRAIN_DEVICE_NOTE = "8GB · ~16GB RAM · AMP / cudnn.benchmark defaults"
OUT_ROOT_TAG = "5060_shallow_net_enhance_three_accpaper"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_2s_hop100"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    # --- 吞吐：8GB 显存稳妥默认；轻模型可 --batch-train 256 ---
    batch_train: int = 128
    batch_eval: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "2s-hop100ms-balbatch-accpaper-openbmi "
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    openbmi_only: bool = True
    # --- DataLoader / 设备 ---
    num_workers: int = _default_num_workers()
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking: bool = True
    torch_num_threads: int = 6
    cudnn_benchmark: bool = True
    deterministic: bool = False
    use_amp: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
