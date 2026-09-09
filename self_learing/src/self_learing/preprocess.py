from __future__ import annotations
import numpy as np

def zcore_per_channel(seg_8xT:np.ndarray,eps:float=1e-8):
    x=np.asarray(seg_8xT, dtype=np.float64)
    mean=x.mean(axis=1,keepdims=True)
    std=x.std(axis=1,keepdims=True)
    return (x-mean)/(std+eps)

def baseline_correct(
        seg_8xT:np.asarray,baseline_8xTb:np.ndarray
)->np.ndarray:
    """段级：每通道减去 baseline 时段均值。"""
    base=baseline_8xTb.mean(axis=1,keepdims=True)
    return np.asarray(seg_8xT,dtype=np.float64)-base
