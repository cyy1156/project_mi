"""baselines_2s_hop100_accpaper 共用超参：Val Acc_paper 早停；train batch balance；无 RAP；仅 T。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "bci2a_2s_hop100"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 18
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = "2s-hop100ms-balbatch-accpaper-T-only"
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    bci2a_T_only: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
