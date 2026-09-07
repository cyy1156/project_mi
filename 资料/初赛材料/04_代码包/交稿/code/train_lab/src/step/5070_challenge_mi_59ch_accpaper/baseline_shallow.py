"""ShallowFBCSPNet · Exp34 轨 A · 59ch · 5070。"""
from __future__ import annotations

import torch.nn as nn
from braindecode.models import ShallowFBCSPNet

from task_runner import run_baseline_main


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


if __name__ == "__main__":
    run_baseline_main(
        model_name="shallow",
        build_model=build_model,
        structure_note="ShallowFBCSPNet · challenge 59ch · Exp34 · 5070",
        extra_meta={"experiment": 34, "device": "5070", "track": "A59", "seed": 42},
    )
