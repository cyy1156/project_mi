"""ShallowFBCSPNet · Tw=3s hop=100ms Acc_paper（实验 21 · 5070）。

相对正式 2s 基线：仅 n_times=750 / data_tag=openbmi_3s_hop100；协议其余冻结。
相对实验 20：仅机位/out 前缀 → 5070。
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
        structure_note="ShallowFBCSPNet（braindecode 默认）· Tw=3s hop=100ms · 实验21·5070",
        extra_meta={
            "shallow": {"backbone": "ShallowFBCSPNet"},
            "accpaper": True,
            "experiment": 21,
            "device": "5070",
            "win_sec": 3.0,
            "hop_sec": 0.1,
        },
    )
