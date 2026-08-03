"""GCBNet 1s 离线：1s 窗 bandpower；Val BalAcc + batch balance。"""
from __future__ import annotations

import torch.nn as nn

from feat_bandpower import raw_to_bandpower
from load_external import load_baselines_single
from task_runner import run_baseline_main

_old = load_baselines_single("baseline_gcbnet.py")


def build_model(n_electrodes: int, n_feats: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return _old.build_model(n_electrodes, n_feats, n_outputs, drop_prob)


if __name__ == "__main__":
    run_baseline_main(
        model_name="gcbnet",
        build_model=build_model,
        input_kind="feat",
        structure_note="GCBNet(k=2, layers=[128]) + 1s bandpower",
        prepare_X=raw_to_bandpower,
        extra_meta={"gcbnet": {"backbone": "GCBNet", "k": 2}},
    )
