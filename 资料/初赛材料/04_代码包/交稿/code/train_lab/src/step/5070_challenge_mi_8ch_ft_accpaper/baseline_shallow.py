from __future__ import annotations
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet
from task_runner import run_baseline_main

def build_model(n_chans, n_times, n_outputs, drop_prob):
    return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob)

if __name__ == "__main__":
    run_baseline_main(model_name="shallow", build_model=build_model,
                      structure_note="Shallow · B8 OpenBMI-ft · Exp34 · 5070",
                      extra_meta={"track": "B8"})
