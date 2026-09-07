"""从 baselines_2s_hop100 只读加载 build_model / prepare_X（不修改原包）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch.nn as nn

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
HOP100 = STEP / "baselines_2s_hop100"

# hop100 侧依赖（feat_bandpower / raw_time / load_external）
for p in (str(HOP100), str(STEP), str(STEP.parents[1] / "preprocess_lab")):
    # STEP.parents[1] is train_lab? HERE=trialmaj, parent=step, parents[1]=src, parents[2]=train_lab, parents[3]=code
    pass

CODE_ROOT = HERE.parents[3]
PRE_ROOT = CODE_ROOT / "preprocess_lab"
for p in (HOP100, STEP, PRE_ROOT, STEP / "baselines_single"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _load_hop100_module(name: str):
    path = HOP100 / f"baseline_{name}.py"
    spec = importlib.util.spec_from_file_location(f"_hop100_reeval_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    # 避免执行 if __name__ == main（exec_module 不会进 main）
    spec.loader.exec_module(mod)
    return mod


# model -> (input_kind, prepare name)
# time: ArrayTaskDataset squeeze; feat: bandpower; raw: squeeze_raw
MODEL_SPEC: dict[str, tuple[str, str | None]] = {
    "eegnet": ("time", None),
    "shallow": ("time", None),
    "deep": ("time", None),
    "eegtcnet": ("time", None),
    "conformer": ("time", None),
    "dbn": ("feat", "bandpower"),
    "gcbnet": ("feat", "bandpower"),
    "dgcnn": ("feat", "bandpower"),
    "dbn_raw": ("raw", "squeeze"),
    "gcbnet_raw": ("raw", "squeeze"),
    "dgcnn_raw": ("raw", "squeeze"),
}


def get_prepare_X(name: str) -> Callable[[np.ndarray], np.ndarray] | None:
    kind = MODEL_SPEC[name][1]
    if kind is None:
        return None
    if kind == "bandpower":
        from feat_bandpower import raw_to_bandpower

        return raw_to_bandpower
    if kind == "squeeze":
        from raw_time import squeeze_raw_2s

        return squeeze_raw_2s
    raise ValueError(kind)


def get_build_model(name: str) -> Callable[..., nn.Module]:
    mod = _load_hop100_module(name)
    return mod.build_model


def get_input_kind(name: str) -> str:
    return MODEL_SPEC[name][0]
