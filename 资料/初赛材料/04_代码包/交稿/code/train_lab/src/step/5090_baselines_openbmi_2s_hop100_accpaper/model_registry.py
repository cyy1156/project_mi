"""本包全 11 模型（与 baselines_2s_hop100_accpaper 同名单）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch.nn as nn

HERE = Path(__file__).resolve().parent

MODEL_SPEC: dict[str, tuple[str, str | None]] = {
    "shallow": ("time", None),
    "deep": ("time", None),
    "conformer": ("time", None),
    "eegnet": ("time", None),
    "eegtcnet": ("time", None),
    "gcbnet": ("feat", "bandpower"),
    "dgcnn": ("feat", "bandpower"),
    "dbn": ("feat", "bandpower"),
    "dbn_raw": ("feat", "squeeze"),
    "gcbnet_raw": ("feat", "squeeze"),
    "dgcnn_raw": ("feat", "squeeze"),
}

ALL_MODELS = tuple(MODEL_SPEC.keys())


def _load_local_baseline(name: str):
    path = HERE / f"baseline_{name}.py"
    spec = importlib.util.spec_from_file_location(f"openbmi_accpaper_baseline_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_prepare_X(name: str) -> Callable[[np.ndarray], np.ndarray] | None:
    kind = MODEL_SPEC[name][1]
    if kind is None:
        return None
    import _hop100_path  # noqa: F401

    if kind == "bandpower":
        from feat_bandpower import raw_to_bandpower

        return raw_to_bandpower
    if kind == "squeeze":
        from raw_time_openbmi import squeeze_raw_2s

        return squeeze_raw_2s
    raise ValueError(kind)


def get_build_model(name: str) -> Callable[..., nn.Module]:
    return _load_local_baseline(name).build_model


def get_input_kind(name: str) -> str:
    return MODEL_SPEC[name][0]


def get_structure_note(name: str) -> str:
    if name == "deep":
        return "Deep4Net-compat pool_time_length/stride=1"
    kind = MODEL_SPEC[name][0]
    return f"baseline_{name}.py input_kind={kind}"
