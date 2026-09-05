"""S0 · 手写 ShallowFBCSPNet（attn=None）· 方案 18 审计旁路。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SELF_MODEL = REPO / "self_model"
if str(SELF_MODEL) not in sys.path:
    sys.path.insert(0, str(SELF_MODEL))

from shallowfbcsp import ShallowFBCSPNet  # noqa: E402  # pyright: ignore[reportMissingImports]

from task_runner import run_baseline_main


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
        attn=None,
    )


if __name__ == "__main__":
    run_baseline_main(
        model_name="shallow_S0_self",
        build_model=build_model,
        input_kind="time",
        structure_note="ShallowFBCSPNet · self_model/shallowfbcsp · attn=None · 方案18 S0",
        extra_meta={
            "scheme18": {"arm": "S0", "backbone": "self_model.shallowfbcsp"},
            "accpaper": True,
            "bypass": True,
        },
    )
