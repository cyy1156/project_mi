"""全基线共用训练超参（改这里 = 所有 baseline_*.py 一起变）。

seed：同时用于（1）被试五折划分；（2）各 baseline 的 seed_everything / DataLoader generator。
改 seed 会同时改变划分与训练随机性；默认同环境可复现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "merged_2s"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42  # 划分 + 全局锁种；每折再建模前用 seed+fold
    max_epochs: int = 300
    patience: int = 18
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
