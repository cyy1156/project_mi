"""本包可见的 feat_bandpower：转发到 baselines_2s_hop100。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_HOP100 = Path(__file__).resolve().parent.parent / "baselines_2s_hop100"
_MOD_NAME = "_openbmi_fwd_hop100_feat_bandpower"


def _load() -> ModuleType:
    cached = sys.modules.get(_MOD_NAME)
    if cached is not None:
        return cached
    path = _HOP100 / "feat_bandpower.py"
    spec = importlib.util.spec_from_file_location(_MOD_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    hop = str(_HOP100)
    if hop not in sys.path:
        sys.path.append(hop)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load()
raw_to_bandpower = _mod.raw_to_bandpower

__all__ = ("raw_to_bandpower",)
