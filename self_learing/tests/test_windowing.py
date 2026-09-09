from __future__ import annotations

import numpy as np
from self_learing.src.self_learing.preprocess import zcore_per_channel
from self_learing.src.self_learing.windowing import WindowSpec,iter_windows_specs,slice_window_8xT

def test_n_windows()->None:
    specs