from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class WindowSpec:
    t_start: float
    t_end: float

def iter_windows_specs(
    task_s: float=4.0,
    win_s: float=3.0,
    hop_s:float=0.1,
)->List[WindowSpec]:
    specs: List[WindowSpec]=[]
    t0=0.0
    while t0+win_s <= task_s+1e-9:
        
