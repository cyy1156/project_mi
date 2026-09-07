"""DBN + TemporalEncoder：1s 原始时域；Val BalAcc + batch balance。"""
from __future__ import annotations

import torch.nn as nn

from load_external import load_selfdev
from raw_time import squeeze_raw_1s
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
        structure_note="TemporalEncoder(D=64) + DBN；1s 原始时域 (B,8,250)",
        prepare_X=squeeze_raw_1s,
        extra_meta={
            "dbn_raw": {
                "backbone": "DBNRaw",
                "node_dim": int(_mod.NODE_DIM),
                "source": "Self_development_model/TeporalEncoder_dbn.py",
            }
        },
    )
