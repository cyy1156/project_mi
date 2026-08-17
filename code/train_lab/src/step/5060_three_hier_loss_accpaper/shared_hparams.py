"""Scheme 16 · 5060 · OpenBMI Acc_paper · shallow + hierarchical Three loss.

本机 RTX 5060 / ~16GB：低内存默认（workers=0、关 pin、可保留 pack）。
全量五折请用姊妹包：`5090_three_hier_loss_accpaper/`。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


TRAIN_DEVICE_LABEL = "NVIDIA RTX 5060 Laptop"
TRAIN_DEVICE_NOTE = "16GB · scheme16 hier-loss · low-mem defaults"
OUT_ROOT_TAG = "5060_three_hier_loss_accpaper"


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "openbmi_2s_hop100"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    # 16GB：小 batch，降低 CUDA workspace / 提交内存
    batch_train: int = 64
    batch_eval: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss "
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train "
        "device=5060"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    openbmi_only: bool = True
    # 与方案 14/15 相同：折内 float16 pack（实测 stream_windows 在 16GB 机上更易把物理内存打满）
    stream_windows: bool = False
    # pack 后保留 _cache_*.npy，OOM/重试时可 reuse
    keep_fold_packs: bool = True
    num_workers: int = 0
    # 16GB 机：关 pin，避免折内 float16 pack + 钉扎峰值 OOM
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    non_blocking: bool = False
    torch_num_threads: int = 2
    # 关 benchmark：少占 CUDA 提交内存（16GB 机 Event 2004 常见）
    cudnn_benchmark: bool = False
    deterministic: bool = False
    use_amp: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
