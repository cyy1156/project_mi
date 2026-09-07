"""OpenBMI 包装 hop100.load_external：加载外部 build_model 后恢复本包超参与 sys.path。

hop100 版 load_external 会把 shared_hparams 切到 baselines_2s_hop100，
并可能污染已加载的 task_runner.SHARED，导致缺少 openbmi 字段（如 cudnn_benchmark）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
HOP100 = HERE.parent / "baselines_2s_hop100"

_spec = importlib.util.spec_from_file_location(
    "_hop100_load_external", HOP100 / "load_external.py"
)
_hop = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hop)


def _restore_openbmi_context() -> None:
    here = str(HERE)
    if here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, here)
    for name in ("shared_hparams", "md_fold_detail"):
        sys.modules.pop(name, None)
    import shared_hparams  # noqa: F401
    import md_fold_detail  # noqa: F401
    tr = sys.modules.get("task_runner")
    sh = sys.modules.get("shared_hparams")
    if tr is not None and sh is not None and hasattr(sh, "SHARED"):
        tr.SHARED = sh.SHARED
        if hasattr(sh, "SharedTrainHP"):
            tr.SharedTrainHP = sh.SharedTrainHP
        if hasattr(sh, "shared_as_dict"):
            tr.shared_as_dict = sh.shared_as_dict


def load_baselines_single(name: str) -> ModuleType:
    mod = _hop.load_baselines_single(name)
    _restore_openbmi_context()
    return mod


def load_selfdev(name: str) -> ModuleType:
    mod = _hop.load_selfdev(name)
    _restore_openbmi_context()
    return mod
