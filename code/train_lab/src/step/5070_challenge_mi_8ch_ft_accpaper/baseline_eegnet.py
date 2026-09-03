from __future__ import annotations
import torch.nn as nn
try:
    from braindecode.models import EEGNet
except ImportError:
    from braindecode.models import EEGNetv4 as EEGNet
from task_runner import run_baseline_main

def build_model(n_chans, n_times, n_outputs, drop_prob):
    return EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, F1=8, D=2, F2=16, drop_prob=drop_prob)

if __name__ == "__main__":
    run_baseline_main(model_name="eegnet", build_model=build_model,
                      structure_note="EEGNet · B8 · Exp34 · 5070",
                      extra_meta={"track": "B8"})
