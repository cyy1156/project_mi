"""方案 26 · R1/R2/R3 训练配方超参。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

OUT_ROOT_TAG = "5090_ens_recipe_3s_hop100_accpaper"
SCHEME26_RUNS_TAG = "5090_openbmi_3s_hop100"
TRAIN_DEVICE_LABEL = "NVIDIA RTX 5090"


@dataclass(frozen=True)
class RecipeTrainHP:
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
        "subject_key=openbmi:subjNN device=5090 scheme26"
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
    # recipe
    optimizer: str = "adamw"
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 3
    lr_min: float = 1e-6
    early_stop_tie_eps: float = 0.0005
    use_swa: bool = False
    swa_start_frac: float = 0.5


def recipe_for(arm: str) -> RecipeTrainHP:
    base = RecipeTrainHP()
    if arm == "R1":
        return base
    if arm == "R2":
        return replace(base, use_swa=True)
    if arm == "R3":
        return replace(base, weight_decay=5e-4)
    raise ValueError(f"unknown recipe arm {arm}")


def as_dict(hp: RecipeTrainHP) -> dict:
    return asdict(hp)
