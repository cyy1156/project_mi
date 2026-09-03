from __future__ import annotations
import torch.nn as nn
from braindecode.models import EEGConformer
from shared_hparams import hp_conformer
from task_runner import run_baseline_main

def build_model(n_chans, n_times, n_outputs, drop_prob):
    return EEGConformer(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
        final_fc_length="auto", drop_prob=drop_prob,
        num_layers=2, num_heads=10, att_drop_prob=0.5,
    )

if __name__ == "__main__":
    run_baseline_main(model_name="conformer", build_model=build_model,
                      structure_note="Conformer · B8 · Exp34 · 5070",
                      extra_meta={"track": "B8"},
                      hp=hp_conformer())
