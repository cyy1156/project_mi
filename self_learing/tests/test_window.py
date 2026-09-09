from __future__ import annotations

import numpy as np

from self_learing.preprocess import zcore_per_channel
from self_learing.windowing import WindowSpec,iter_windows_specs,slice_window_8xT

def test_n_windows()->None:
    specs=iter_windows_specs(4.0,3.0,0.1)
    assert len(specs)==11
    assert (specs[0].t_start-0.0)<1e-9
    assert (specs[0].t_end-3.0)<1e-9

def test_shape_750()->None:
    fs=250.0
    T=int(10*fs)
    data=np.random.randn(T,8)
    spec=WindowSpec(0.0,3.0)
    win=slice_window_8xT(data,fs,anchor_t=1.0,spec=spec)
    assert win.shape ==(8,750)
    z=zcore_per_channel(win)
    assert z.shape ==(8,750)