"""Deep4Net · compat pool=1/1 · 游戏全模型微调。"""
from __future__ import annotations

import torch.nn as nn
from braindecode.models import Deep4Net

from task_runner import run_ft_main


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return Deep4Net(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
        pool_time_length=1,
        pool_time_stride=1,
    )


if __name__ == "__main__":
    run_ft_main(
        model_name="deep",
        build_model=build_model,
        structure_note="Deep4Net-compat（pool=1/1）；full_model FT",
        extra_meta={
            "deep": {
                "backbone": "Deep4Net",
                "compat": "pool_time_length=1,pool_time_stride=1",
            }
        },
    )
