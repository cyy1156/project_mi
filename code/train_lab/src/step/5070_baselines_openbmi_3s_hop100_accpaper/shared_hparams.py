"""OpenBMI Acc_paper · 3s/hop100 · 仅 Shallow（实验 21 · 5070）。

相对 5060 实验 20 包：同协议 Tw=3s；机位/out 前缀改为 5070。
超参对齐 `5070_本机配置与实验计划.md`：batch 128/256 · AMP。
禁止写入 `out/5060_*`。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _default_num_workers() -> int:
    # RTX 5070 Laptop · 提速默认；OOM / 内存紧时 CLI 改 --num-workers 0
    return 2


TRAIN_DEVICE_LABEL = "NVIDIA GeForce RTX 5070 Laptop"
TRAIN_DEVICE_NOTE = "8GB VRAM · experiment 21 · Tw=3s hop=100ms Acc_paper shallow-only · batch 128/256"
OUT_ROOT_TAG = "5070_baseline_openbmi_3s_hop100_accpaper"
SCHEME21_RUNS_TAG = "5070_openbmi_3s_hop100"


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
        "subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 750
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


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
