"""A2：A1 输入 + 试次质量加权 CE（需先跑 export_trial_quality.py）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet

from channel_fe import prepare_laterality_X
from task_runner import run_baseline_main

WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "out"
    / "_fe_cache"
    / "openbmi_2s_hop100_trial_quality_weights.npy"
)


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


if __name__ == "__main__":
    if not WEIGHTS_PATH.is_file():
        raise SystemExit(
            f"缺少质量权重：{WEIGHTS_PATH}\n请先运行：python export_trial_quality.py"
        )
    w = np.load(WEIGHTS_PATH)
    run_baseline_main(
        model_name="shallow_a2_lat_qw",
        build_model=build_model,
        input_kind="time",
        structure_note="ShallowFBCSPNet A2 A1-ch + trial quality weighted CE",
        prepare_X=prepare_laterality_X,
        sample_weights=w,
        extra_meta={
            "shallow_mi_feat": {
                "arm": "A2",
                "n_chans": 10,
                "weights": str(WEIGHTS_PATH),
            },
            "accpaper": True,
            "bypass": True,
        },
    )
