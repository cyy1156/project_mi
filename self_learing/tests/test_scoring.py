from __future__ import annotations

import numpy as np

from self_learing.readout import causal_smooth_preds
from self_learing.srcoring import MiTrialTracker

def test_causal_lookback2_hand() -> None:
    p0=np.asarray([0.8,0.1,0.1])
    p1=np.asarray([0.1,0.8,0.1])
    p2=np.array([0.8,0.1,0.1])
    preds=causal_smooth_preds([p0,p1,p2],lookback=2)
    assert preds[0]==0
    assert len(preds)==3

def test_rest_half_point() -> None:
    tr=MiTrialTracker("rest")
    for _ in range(3):
        tr.add_window([0.9,0.05,0.05])
    out =tr.finalize()
    assert out["ok"] is True
    assert out["score"]==0.5