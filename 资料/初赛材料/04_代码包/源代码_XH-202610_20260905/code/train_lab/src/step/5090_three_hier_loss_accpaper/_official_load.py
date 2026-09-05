"""Load Acc_paper train loop (pack/reuse patches live in 5060_baselines).

5090 方案16 与 5060 共用同一套 fold pack 实现，仅 out / hparams / 设备标签不同。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# 训练环与 pack 低峰改动在 5060_baselines；勿分叉两份 materialize
OFFICIAL = Path(__file__).resolve().parent.parent / "5060_baselines_openbmi_2s_hop100_accpaper"
HOP100 = Path(__file__).resolve().parent.parent / "baselines_2s_hop100"
STEP = Path(__file__).resolve().parent.parent
# .../train_lab/src/step/<pkg>/file.py → parents[4] == code/
PRE = Path(__file__).resolve().parents[4] / "preprocess_lab"

for p in (str(STEP), str(PRE), str(HOP100), str(OFFICIAL)):
    if p not in sys.path:
        sys.path.append(p)


def load_official(mod_name: str):
    path = OFFICIAL / f"{mod_name}.py"
    key = f"_hier5090_official_{mod_name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
