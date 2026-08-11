"""ShallowFBCSPNet S0：默认结构（复现锚点）。

S0 = braindecode ShallowFBCSPNet 原版，不做任何结构修改。
后续 S1/S2/... 变体以此为基线对照，隔离结构增强的增益。
"""
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
        model_name="shallow_s0",
        build_model=build_model,
        input_kind="time",
        structure_note="ShallowFBCSPNet（braindecode 默认；S0 复现锚点）",
        extra_meta={
            "shallow": {"backbone": "ShallowFBCSPNet", "variant": "S0_default"},
            "accpaper": True,
        },
    )
