"""Deep4Net 1s 离线：原结构（n_times=250 时库可能自动缩核）；Val BalAcc + batch balance。"""
from __future__ import annotations

import torch.nn as nn
from braindecode.models import Deep4Net

from task_runner import run_baseline_main


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return Deep4Net(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


if __name__ == "__main__":
    run_baseline_main(
        model_name="deep",
        build_model=build_model,
        input_kind="time",
        structure_note="Deep4Net（braindecode 默认；250 点可能自动缩核）",
        extra_meta={"deep": {"backbone": "Deep4Net"}},
    )
