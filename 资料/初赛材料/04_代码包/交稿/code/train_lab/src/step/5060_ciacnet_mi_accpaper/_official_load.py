"""Load official hop100 Acc_paper modules by file path (avoid name clash)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parent.parent / "5060_baselines_openbmi_2s_hop100_accpaper"
HOP100 = Path(__file__).resolve().parent.parent / "baselines_2s_hop100"
STEP = Path(__file__).resolve().parent.parent
PRE = Path(__file__).resolve().parents[3] / "preprocess_lab"

for p in (str(STEP), str(PRE), str(HOP100), str(OFFICIAL)):
    if p not in sys.path:
        sys.path.append(p)


def load_official(mod_name: str):
    path = OFFICIAL / f"{mod_name}.py"
    # unique sys.modules key so we don't clash with local stubs
    key = f"_ciacnet_official_{mod_name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
