"""Deep4Net 游戏伪在线：compat pool=1/1（与 accpaper 一致）；权重=balbatch_accpaper。"""
from __future__ import annotations

import torch.nn as nn
from braindecode.models import Deep4Net

from task_runner import run_pseudo_online_main


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
    run_pseudo_online_main(
        model_name="deep",
        build_model=build_model,
        structure_note="Deep4Net-compat（pool=1/1）",
        extra_meta={
            "deep": {
                "backbone": "Deep4Net",
                "compat": "pool_time_length=1,pool_time_stride=1",
            }
        },
    )
