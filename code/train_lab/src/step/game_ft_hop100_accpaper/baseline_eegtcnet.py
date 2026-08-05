"""EEGTCNet · 游戏全模型微调。"""
from __future__ import annotations

import torch.nn as nn
from braindecode.models import EEGTCNet

from task_runner import run_ft_main


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return EEGTCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


if __name__ == "__main__":
    run_ft_main(
        model_name="eegtcnet",
        build_model=build_model,
        structure_note="EEGTCNet（braindecode 默认）；full_model FT",
        extra_meta={"eegtcnet": {"backbone": "EEGTCNet"}},
    )
