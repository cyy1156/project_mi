"""ShallowFBCSPNet 2s/hop100 离线：原结构；Val BalAcc + batch balance。"""
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
        structure_note="ShallowFBCSPNet（braindecode 默认）",
        extra_meta={"shallow": {"backbone": "ShallowFBCSPNet"}},
    )
