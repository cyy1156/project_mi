"""EEGNet 游戏伪在线：结构同 accpaper；权重=balbatch_accpaper。"""
from __future__ import annotations

import torch.nn as nn

try:
    from braindecode.models import EEGNet
except ImportError:
    from braindecode.models import EEGNetv4 as EEGNet

from task_runner import run_pseudo_online_main

EEGNET_F1, EEGNET_D, EEGNET_F2 = 8, 2, 16


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return EEGNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        F1=EEGNET_F1,
        D=EEGNET_D,
        F2=EEGNET_F2,
        drop_prob=drop_prob,
    )


if __name__ == "__main__":
    run_pseudo_online_main(
        model_name="eegnet",
        build_model=build_model,
        structure_note=f"EEGNet F1={EEGNET_F1}, D={EEGNET_D}, F2={EEGNET_F2}",
        extra_meta={"eegnet": {"F1": EEGNET_F1, "D": EEGNET_D, "F2": EEGNET_F2}},
    )
