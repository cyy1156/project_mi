from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.io.load_bci2a_mat import load_bci2a_mat
from src.eeg_types import ContinuousEEG

LoaderFn = Callable[[Path], list[ContinuousEEG]|ContinuousEEG]

LOADERS:dict[str, LoaderFn] = {
    "load_bci2a_mat": load_bci2a_mat,
    # 以后:
    # "load_openbci_csv": load_openbci_csv,
}

def get_loader(name: str) -> LoaderFn:
    if name not in LOADERS:
       raise KeyError(f"未知 loader: {name}，可选: {list(LOADERS)}")
    return LOADERS[name]

def as_run_list(obj:list[ContinuousEEG] | ContinuousEEG)->list[ContinuousEEG]:
    if isinstance(obj, list):
        return obj
    return [obj]