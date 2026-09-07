"""Re-export Acc_paper float16 squeeze（供 IDE 解析 + 运行时加载）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parent.parent
    / "5060_baselines_openbmi_2s_hop100_accpaper"
    / "raw_time_openbmi.py"
)
_spec = importlib.util.spec_from_file_location("_scheme16_5090_raw_time_openbmi", _PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

squeeze_raw_2s_openbmi = _mod.squeeze_raw_2s_openbmi

__all__ = ["squeeze_raw_2s_openbmi"]
