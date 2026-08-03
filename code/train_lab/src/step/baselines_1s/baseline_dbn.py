"""DBN 1s 离线：1s 窗 bandpower；Val BalAcc + batch balance。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch.nn as nn

from feat_bandpower import raw_to_bandpower
from task_runner import run_baseline_main

_OLD = Path(__file__).resolve().parent.parent / "baselines_single" / "baseline_dbn.py"
_spec = importlib.util.spec_from_file_location("_old_baseline_dbn_1s", _OLD)
_old = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_old)


def build_model(n_electrodes: int, n_feats: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return _old.build_model(n_electrodes, n_feats, n_outputs, drop_prob)


if __name__ == "__main__":
    run_baseline_main(
        model_name="dbn",
        build_model=build_model,
        input_kind="feat",
        structure_note="DBN + 1s μ/β log bandpower (N,8,2)",
        prepare_X=raw_to_bandpower,
        extra_meta={"dbn": {"backbone": "DBN", "bands": [[8, 13], [13, 30]]}},
    )
