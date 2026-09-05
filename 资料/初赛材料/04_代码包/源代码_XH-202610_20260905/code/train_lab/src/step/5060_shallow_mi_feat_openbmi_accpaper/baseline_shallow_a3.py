"""A3：A1 偏侧通道 + C3/C4 Mu(8–13Hz) 包络 → n_chans=12。"""
from __future__ import annotations

import torch.nn as nn
from braindecode.models import ShallowFBCSPNet

from channel_fe import prepare_laterality_mu_X
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
        model_name="shallow_a3_lat_mu",
        build_model=build_model,
        input_kind="time",
        structure_note="ShallowFBCSPNet A3 laterality+Mu envelope (12ch)",
        prepare_X=prepare_laterality_mu_X,
        extra_meta={
            "shallow_mi_feat": {
                "arm": "A3",
                "n_chans": 12,
                "extra_ch": ["C3-C4", "CP3-CP4", "env_C3_mu", "env_C4_mu"],
            },
            "accpaper": True,
            "bypass": True,
        },
    )
