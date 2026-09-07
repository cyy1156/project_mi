"""A0：8 导 raw 复现锚点（旁路；不覆盖正式 shallow out）。"""
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
        model_name="shallow_a0",
        build_model=build_model,
        input_kind="time",
        structure_note="ShallowFBCSPNet A0 raw8（旁路复现锚点）",
        extra_meta={
            "shallow_mi_feat": {"arm": "A0", "n_chans": 8},
            "accpaper": True,
            "bypass": True,
        },
    )
