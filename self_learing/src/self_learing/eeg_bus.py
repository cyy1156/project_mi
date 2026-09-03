"""函数调用的类"""

from __future__ import annotations

from typing import Callable,List

import numpy as np

Subscriber=Callable[[float,np.ndarray],None]

class EegBus:
    def __init__(self)->None:
        self._subs:List[Subscriber]=[]


    def subscribe(self,fn:Subscriber)->None:
        self._subs.append(fn)

    def publish(self,t:float,x:np.ndarray)->None:

        for fn in list(self._subs):
            fn(float(t),np.asarray(x,dtype=np.float64))
