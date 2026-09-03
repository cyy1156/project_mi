from __future__ import annotations

from email import header
from pathlib import Path
from typing import TextIO,List
import numpy as np


class CsvRecoder:
    def __init__(self,path:Path,channle_names:list[str])->None:
        self.path= Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.channle_names=list(channle_names)
        self._f:TextIO = self.path.open("w",encoding="utf-8",newline="\n")
        header=["t"]+self.channle_names
        self._f.write(",".join(header)+"\n")


    def on_sample(self,t:float,x:np.ndarray)->None:
        x=np.asarray(x,dtype=np.float64).reshape(-1)
        cols=[f"{float(t):.6f}"]+[f"float(v):.8f"for v in x]
        self._f.write(",".join(cols)+"\n")

    def close(self)->None:
        self._f.flush()
        self._f.close()

