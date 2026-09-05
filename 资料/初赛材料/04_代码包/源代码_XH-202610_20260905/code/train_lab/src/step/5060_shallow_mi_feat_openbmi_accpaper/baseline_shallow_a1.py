"""A1：8 导 + C3−C4 / CP3−CP4 偏侧差模通道 → n_chans=10。"""
from __future__ import annotations

import torch.nn as nn
from braindecode.models import ShallowFBCSPNet

from channel_fe import prepare_laterality_X
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
        model_name="shallow_a1_lat",
        build_model=build_model,
        input_kind="time",
        structure_note="ShallowFBCSPNet A1 laterality +2ch (C3-C4, CP3-CP4)",
        prepare_X=prepare_laterality_X,
        extra_meta={
            "shallow_mi_feat": {
                "arm": "A1",
                "n_chans": 10,
                "extra_ch": ["C3-C4", "CP3-CP4"],
            },
            "accpaper": True,
            "bypass": True,
        },
    )
