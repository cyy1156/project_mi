"""Deep4Net 2s/hop100 离线：对齐 1s 自动缩核形态（非满血 stride-3）；Val BalAcc + batch balance。

T=500 时 braindecode 会启用完整 pool stride=3×4，易塌成全预测 Rest（Spec=1/Rec=0）。
1s（T=250）因短于最小长度会自动缩池化；此处显式 pool_time_length/stride=1 以对齐该实际结构。
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
        # 对齐 baselines_1s 在 T=250 下的自动缩核结果（满血 Deep4 在本配方下不可训）
        pool_time_length=1,
        pool_time_stride=1,
    )


if __name__ == "__main__":
    run_baseline_main(
        model_name="deep",
        build_model=build_model,
        input_kind="time",
        structure_note="Deep4Net-compat（pool=1/1，对齐1s自动缩核；非满血stride3）",
        extra_meta={
            "deep": {
                "backbone": "Deep4Net",
                "compat": "pool_time_length=1,pool_time_stride=1",
                "note": "align_1s_auto_shrink",
            }
        },
    )
