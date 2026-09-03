from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np




class RingBuffer:
    def __init__(self,n_ch:int=8,fs:float=250.0,capacity_s:float=30)->None:
        self.n_ch = int(n_ch)
        self.fs = float(fs)
        self.maxlen= max(1,int(fs*capacity_s))
        #构造缓存区

        self._q:Deque[Tuple[float,np.ndarray]] = deque(maxlen=self.maxlen)

    def push(self,t:float,x:np.ndarray)->None:
        x=np.array(x,dtype=np.float64).reshape(-1)
        if x.shape[0]!=self.n_ch:
            raise ValueError(f"期望通道{self.n_ch}，收到通道{x.shape[0]}")
        self._q.append((float(t),x.copy()))


    def __len__(self) ->int:
        return len(self._q)

    def get_latest(self,n:int) -> Tuple[float,np.ndarray]:
        """返回最近 n 个：times shape (n,)；data shape (n, 8)。"""
        n=int(n)
        if n<=0 or len(self._q)<n:
            raise ValueError("缓存不够")

        items =list(self._q)[-n:]
        times=np.array([t for t,_ in items],dtype=np.float64)
        data=np.stack([x for _,x in items],axis=0)

        return times,data




