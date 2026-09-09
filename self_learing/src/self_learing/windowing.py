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
        specs.append(WindowSpec(t0,t0+win_s))
        t0+=hop_s
    return specs

def slice_window_8xT(
        data_tx8:np.ndarray,
        fs:float,
        anchor_t:float,
        spec:WindowSpec,
)->np.ndarray:
    """data_tx8: shape (T, 8)，按绝对时间从 anchor 起切。

        返回 (8, n_samp)。
     """
    data =np.array(data_tx8,dtype=np.float64)
    assert data.ndim ==2 and data.shape[1]==8
    i0=int(round((anchor_t+spec.t_start)*fs))
    i1=int(round((anchor_t+spec.t_end)*fs))
    seg=data[i0:i1]# (n, 8)
    return seg.T.copy()#(8,n)



