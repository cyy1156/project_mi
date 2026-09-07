"""ShallowFBCSPNet · Tw=3s hop=100ms Acc_paper（方案 24 · 5090）。

相对 5070 实验 21：batch 256/512 · out 前缀 5090_alg_incr_3s。
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
        model_name="shallow",
        build_model=build_model,
        input_kind="time",
        structure_note="ShallowFBCSPNet · Tw=3s hop=100ms · 方案24·5090",
        extra_meta={
            "shallow": {"backbone": "ShallowFBCSPNet"},
            "accpaper": True,
            "experiment": 24,
            "device": "5090",
            "win_sec": 3.0,
            "hop_sec": 0.1,
        },
    )
