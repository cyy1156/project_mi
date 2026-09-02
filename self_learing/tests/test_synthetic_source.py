import sys
from pathlib import Path

# 让 Python 能 import 到 src/self_learing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from self_learing.channles import CHANNEL_LABELS, N_CH
from self_learing.source_synthetic import SyntheticSource


def test_iter_n_count_and_shape():
    src = SyntheticSource(fs=250, n_ch=8, seed=0)
    xs = list(src.iter_n(N_CH))  # iter_n 必须告诉它要几个样本
    assert len(xs) == N_CH
    t0, x0 = xs[0]
    assert isinstance(t0, float)       # t 是时间
    assert isinstance(x0, np.ndarray)  # x 是数组
    assert x0.shape == (N_CH,)
    assert x0.dtype == np.float64

def test_time_spacing():
    src =SyntheticSource(fs=250,seed=1)
    xs =list(src.iter_n(3,start_t=1.0))
    assert abs(xs[0][0]-1.0)<1e-9
    assert abs(xs[1][0]-(1.0+1.0/250))<1e-9

def test_channel_names_match_openbmi_order():
    assert CHANNEL_LABELS == [
        "FC3",
        "C3",
        "CP3",
        "CZ",
        "CPZ",
        "FC4",
        "C4",
        "CP4",
    ]
    assert N_CH == 8


def test_start_stop_flags():
    src = SyntheticSource()
    src.start()
    assert src._running is True
    src.stop()
    assert src._running is False

