"""本包可见的 load_external：转发到 baselines_2s_hop100，并恢复本包 shared_hparams。

hop100 的 load_sibling_module 会把 sys.modules['shared_hparams'] 换回 hop100 版，
从而冲掉本包带 torch_num_threads / use_amp 等字段的 SharedTrainHP。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
_HOP100 = HERE.parent / "baselines_2s_hop100"
_MOD_NAME = "_openbmi_fwd_hop100_load_external"
_PIN = ("shared_hparams", "md_fold_detail")


def _load() -> ModuleType:
    cached = sys.modules.get(_MOD_NAME)
    if cached is not None:
        return cached
    path = _HOP100 / "load_external.py"
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


def _restore_openbmi_locals() -> None:
    """强制本包 shared_hparams / md_fold_detail 回到 openbmi 目录版本。"""
    here = str(HERE)
    if here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, here)
    for k in _PIN:
        sys.modules.pop(k, None)
    sh = importlib.import_module("shared_hparams")
    try:
        importlib.import_module("md_fold_detail")
    except Exception:
        pass
    tr = sys.modules.get("task_runner")
    if tr is not None and hasattr(sh, "SHARED"):
        tr.SHARED = sh.SHARED
        if hasattr(sh, "shared_as_dict"):
            tr.shared_as_dict = sh.shared_as_dict
        if hasattr(sh, "SharedTrainHP"):
            tr.SharedTrainHP = sh.SharedTrainHP


_fwd = _load()
load_sibling_module = _fwd.load_sibling_module


def load_baselines_single(name: str) -> ModuleType:
    mod = _fwd.load_baselines_single(name)
    _restore_openbmi_locals()
    return mod


def load_selfdev(name: str) -> ModuleType:
    mod = _fwd.load_selfdev(name)
    _restore_openbmi_locals()
    return mod


__all__ = ("load_sibling_module", "load_baselines_single", "load_selfdev")
