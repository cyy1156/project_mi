"""EEGConformer · Exp34 轨 A · 59ch · 5070（更小 batch）。"""
from __future__ import annotations

import torch.nn as nn
from braindecode.models import EEGConformer

from shared_hparams import hp_for_conformer
from task_runner import run_baseline_main

CONFORMER_NUM_LAYERS = 2
CONFORMER_NUM_HEADS = 10
CONFORMER_ATT_DROP = 0.5


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return EEGConformer(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        final_fc_length="auto",
        drop_prob=drop_prob,
        num_layers=CONFORMER_NUM_LAYERS,
        num_heads=CONFORMER_NUM_HEADS,
        att_drop_prob=CONFORMER_ATT_DROP,
    )


if __name__ == "__main__":
    run_baseline_main(
        model_name="conformer",
        build_model=build_model,
        structure_note=(
            f"EEGConformer layers={CONFORMER_NUM_LAYERS} "
            f"heads={CONFORMER_NUM_HEADS} · Exp34 · 5070"
        ),
        extra_meta={"experiment": 34, "device": "5070", "track": "A59"},
        hp=hp_for_conformer(),
    )
