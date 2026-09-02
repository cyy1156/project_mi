
from typing import Protocol,Optional,Iterator,Tuple


import numpy as np
from self_learing.channles import FS_HZ, N_CH
from self_learing.source_base import BaseSource

class SyntheticSource(BaseSource):
    def __init__(self,
                 fs:float=FS_HZ,
                 n_ch:int =N_CH,
                 seed:Optional[int]=0,
    )->None:
        if int(n_ch) != N_CH:
            raise ValueError(f"n_ch必须为{N_CH}，收到{n_ch}")
        self.fs = fs
        self.n_ch = n_ch
        self._rng= np.random.default_rng(seed)
        self._running=False

    def start(self)->None:
        self._running=True

    def stop(self)->None:
        self._running=False


    def iter_n(
            self,
            n:int,
            start_t:float=0.0,
            )->Iterator[Tuple[float,np.ndarray]]:
            n=int(n)

            for i in range(n):
                t =float(start_t)+i/self.fs
                x=self._rng.standard_normal(self.n_ch).astype(np.float64)
                yield t,x

    def sample(self) -> Iterator[Tuple[float,np.ndarray]]:
        if not self._running:
            self._start()

        i=0
        while self._running:
            t=i/self.fs
            x=self._rng.standard_normal(self.n_ch).astype(np.float64)
            yield t,x
            i+=1



