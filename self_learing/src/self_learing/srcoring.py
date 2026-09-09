from __future__ import annotations

from typing import List,Any,Dict

import numpy as np

from self_learing.readout import causal_smooth_preds,majority_vote

IDX_TO_NAME={
    0:"rest",
    1:"left",
    2:"right",
}
NAMES_TO_IDX={v:k for k,v in IDX_TO_NAME.items()}

class MiTrialTracker:
    def __init__(self,true_label:str)->None:
        self.true_label=true_label
        self.probs:List[np.ndarray]=[]

    def add_window(self,p_three)->None:
        self.probs.append(np.array(p_three,dtype=np.float64).reshape(3))

    def finalize(self)->Dict[str,Any]:
        if not self.probs:
            return{"pred":None,"score":0.0,"ok":False}

        pred=causal_smooth_preds(self.probs,lookback=2)
        pred_idex=majority_vote(pred)
        pred_name=IDX_TO_NAME[pred_idex]
        ok = pred_name==self.true_label
        if not ok:
            score=0
        elif self.true_label=="rest":
            score=0.5
        else :
            score=1.0
        return {"pred":pred_name,
                "pred_idex":pred_idex,
                "window_preds":pred,
                "score":score,
                "ok":ok

        }
