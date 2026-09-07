"""baselines_1s 共用超参：离线 1s、原结构、Val BalAcc 早停 + train batch balance。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SharedTrainHP:
    data_tag: str = "bci2a_1s"
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
    # 协议标记（写入 summary / MD）
    protocol: str = "1s-offline-native-arch"
    early_stop: str = "balanced_accuracy"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 250
    no_rap: bool = True


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
