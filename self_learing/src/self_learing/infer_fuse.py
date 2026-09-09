from __future__ import annotations
from typing import Sequence

import numpy as np

def apply_temperature_probs(probs:np.ndarray,temperature:float)->np.ndarray:
    p=np.asarray(probs,dtype=np.float64)
    T=float(temperature)
    if T<=0:
        raise ValueError('Temperature must be greater than 0')
    q=np.power(np.clip(p,0,1.0),1.0/T)
    # p 是 (3,) 一维时只有 axis=0/-1；用 -1 对 (3,) 和 (N,3) 都适用
    return q / q.sum(axis=-1, keepdims=True)


def fuse_member_probs(
        probs_list:Sequence[np.ndarray],
        temperature:Sequence[float],
        weights:Sequence[np.ndarray],
)->np.ndarray:
    assert len(probs_list)==len(weights)==len(temperature)
    acc =np.zeros(3,dtype=np.float64)
    wsum =0.0
    for p,T,w in zip(probs_list,temperature,weights):
        pt=apply_temperature_probs(p,T)
        acc+=float(w)*pt
        wsum+=w
    out=acc/max(wsum,1e-12)
    return out/out.sum()


