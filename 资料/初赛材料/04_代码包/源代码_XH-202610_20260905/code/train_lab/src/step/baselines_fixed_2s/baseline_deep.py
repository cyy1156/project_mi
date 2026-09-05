"""Deep4Net 固定窗 Cue+2~4s：braindecode 默认结构（n_times=500）。

方案：主表用默认 Deep4；若 Spec≈1 塌缩，另开 compat（pool=1/1）消融，见方案 §2。
"""
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
        structure_note="Deep4Net（braindecode 默认；n_times=500；塌缩则改 compat 消融）",
        extra_meta={"deep": {"backbone": "Deep4Net", "compat": False}},
    )
