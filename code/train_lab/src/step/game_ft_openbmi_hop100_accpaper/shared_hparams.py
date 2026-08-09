"""本臂 FT 超参：全模型微调；Val Acc_paper 早停；balbatch；无 RAP（同 02）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SharedTrainHP:
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    protocol: str = (
        "game-ft-openbmi-half-fullmodel-balbatch-accpaper "
        "finetune_mode=full_model freeze_backbone=false head_only=false "
        "init=openbmi channel_remap=game_to_openbmi"
    )
    early_stop: str = "acc_paper"
    train_sampler: str = "balanced_invfreq"
    n_times_expected: int = 500
    no_rap: bool = True
    no_balbatch: bool = False
    finetune_mode: str = "full_model"
    freeze_backbone: bool = False
    head_only: bool = False


SHARED = SharedTrainHP()


def shared_as_dict() -> dict:
    return asdict(SHARED)
