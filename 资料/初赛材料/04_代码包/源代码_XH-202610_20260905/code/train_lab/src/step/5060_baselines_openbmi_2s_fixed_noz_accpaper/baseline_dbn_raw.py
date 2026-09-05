"""DBN + TemporalEncoder：OpenBMI 2s/hop100 Acc_paper 原始时域。"""
from __future__ import annotations

import torch.nn as nn

import _hop100_path  # noqa: F401
from load_external import load_selfdev
from raw_time_openbmi import squeeze_raw_2s_openbmi
from task_runner import run_baseline_main

_mod = load_selfdev("TeporalEncoder_dbn.py")


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    _ = n_chans
    return _mod.DBNRaw(
        n_times=int(n_times),
        node_dim=int(_mod.NODE_DIM),
        n_outputs=n_outputs,
        drop_prob=drop_prob,
    )


if __name__ == "__main__":
    run_baseline_main(
        model_name="dbn_raw",
        build_model=build_model,
        input_kind="feat",
        structure_note="TemporalEncoder(D=64) + DBN；OpenBMI 原始时域 (B,8,500)",
        prepare_X=squeeze_raw_2s_openbmi,
        extra_meta={
            "dbn_raw": {
                "backbone": "DBNRaw",
                "node_dim": int(_mod.NODE_DIM),
                "source": "Self_development_model/TeporalEncoder_dbn.py",
            },
            "accpaper": True,
        },
    )
