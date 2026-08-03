"""GCBNet + TemporalEncoder：1s 原始时域；Val BalAcc + batch balance。

模型定义来自 Self_development_model/TepmoralEncoder_GCBNet.py（n_times=250）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch.nn as nn

from raw_time import squeeze_raw_1s
from task_runner import run_baseline_main

_SRC = (
    Path(__file__).resolve().parent.parent
    / "Self_development_model"
    / "TepmoralEncoder_GCBNet.py"
)
_spec = importlib.util.spec_from_file_location("_selfdev_gcbnet_raw_1s", _SRC)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


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
        input_kind="feat",
        structure_note="TemporalEncoder(D=64) + GCBNet(k=2)；1s 原始时域 (B,8,250)",
        prepare_X=squeeze_raw_1s,
        extra_meta={
            "gcbnet_raw": {
                "backbone": "GCBNetRaw",
                "node_dim": int(_mod.NODE_DIM),
                "source": "Self_development_model/TepmoralEncoder_GCBNet.py",
            }
        },
    )
