from __future__ import annotations

import numpy as np

class FakeModel:
    def __init__(self,p_three=(0.5,0.3,0.2)):
        p = np.asarray(p_three,dtype=np.float64).reshape(3)
        p = p/p.sum()
        self.p = p

    def predict_proba(self,win_8x750:np.ndarray)->np.ndarray:
        _=win_8x750
        return self.p

