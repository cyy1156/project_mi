from __future__ import annotations

from typing import List,Sequence

import numpy as np

def causal_smooth_preds(
        probs:Sequence[np.ndarray],lookback:int =2
)->List[int]:
    probs =[np.asarray(p,dtype=np.float64).reshape(3) for p in probs]
    preds:List[int]=[]
    for i in range(len(probs)):
        lo=max(0,i-lookback)
        mean_p=np.mean(probs[lo:i],axis=0)
        preds.append(int(mean_p.argmax()))
    return preds

def majority_vote(preds:Sequence[int]) -> int:
    vals,counts = np.unique(np.asarray(preds,dtype=int),return_counts=True)
    return int(vals[np.argmax(counts)])
