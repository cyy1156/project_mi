"""GCBNet + TemporalEncoder：fixed2s 原始时域；Val BalAcc + batch balance。"""
from __future__ import annotations

import torch.nn as nn

from load_external import load_selfdev
from raw_time import squeeze_raw_2s
from task_runner import run_baseline_main

_mod = load_selfdev("TepmoralEncoder_GCBNet.py")


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    _ = n_chans
    return _mod.GCBNetRaw(
        n_times=int(n_times),
        node_dim=int(_mod.NODE_DIM),
        n_outputs=n_outputs,
        drop_prob=drop_prob,
        graph_hidden=128,
        relu_is=1,
    )


if __name__ == "__main__":
    run_baseline_main(
        model_name="gcbnet_raw",
        build_model=build_model,
        input_kind="raw",
        structure_note="TemporalEncoder(D=64) + GCBNet(k=2)；fixed2s 原始时域 (B,8,500)",
        prepare_X=squeeze_raw_2s,
        extra_meta={
            "gcbnet_raw": {
                "backbone": "GCBNetRaw",
                "node_dim": int(_mod.NODE_DIM),
                "source": "Self_development_model/TepmoralEncoder_GCBNet.py",
            }
        },
    )
