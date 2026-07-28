"""全基线共用训练超参（改这里 = 所有 baseline_*.py 一起变）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "merged_2s"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 100
    patience: int = 18
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 7e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.55


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
