from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.common.eeg_types import ContinuousEEG
from src.datasets.bci2a.load_mat import load_bci2a_mat
from src.datasets.openbmi.load_mat import load_openbmi_mat
from src.datasets.stieger.load_mat import load_stieger_mat

LoaderFn = Callable[[Path], list[ContinuousEEG]|ContinuousEEG]

LOADERS:dict[str, LoaderFn] = {
    "load_bci2a_mat": load_bci2a_mat,
    "load_openbmi_mat": load_openbmi_mat,
    "load_stieger_mat": load_stieger_mat,  # 返回 list[StiegerTrial]，勿直接塞进 2a preprocess_run
}

def get_loader(name: str) -> LoaderFn:
    if name not in LOADERS:
       raise KeyError(f"未知 loader: {name}，可选: {list(LOADERS)}")
    return LOADERS[name]

def as_run_list(obj:list[ContinuousEEG] | ContinuousEEG)->list[ContinuousEEG]:
    if isinstance(obj, list):
        return obj
    return [obj]