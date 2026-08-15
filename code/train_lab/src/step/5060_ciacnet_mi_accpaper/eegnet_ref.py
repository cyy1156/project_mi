"""EEGNet reference for P-track (same train loop / paper config)."""
from __future__ import annotations

import torch.nn as nn

try:
    from braindecode.models import EEGNet
except ImportError:
    from braindecode.models import EEGNetv4 as EEGNet

# Paper compares against Lawhern EEGNet defaults
EEGNET_F1, EEGNET_D, EEGNET_F2 = 8, 2, 16


def build_eegnet(n_chans: int, n_times: int, n_outputs: int, drop_prob: float = 0.5) -> nn.Module:
    # EEGNet paper default drop 0.5; CIACNet paper uses uniform settings across models —
    # keep braindecode defaults for a faithful "EEGNet" baseline, drop_prob from caller if set.
    return EEGNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        F1=EEGNET_F1,
        D=EEGNET_D,
        F2=EEGNET_F2,
        drop_prob=drop_prob,
    )
