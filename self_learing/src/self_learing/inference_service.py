from __future__ import annotations

from self_learing.infer_fake import FakeModel
from typing import Any,Dict
import numpy as np

class InferenceService:
    def __init__(self,model:FakeModel)->None:
        self.model=model

    def judge_window(self,win_8x750:np.ndarray)->Dict[str,Any]:
        p=np.asarray(self.model.predict_proba(win_8x750),dtype=np.float64)
        p=p/p.sum()
        pred =int (p.argmax())
        return {"pred":pred,"p_three":p}

