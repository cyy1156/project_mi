from __future__ import annotations
from typing import Iterator,Protocol,Tuple

import numpy as np

class BaseSource(Protocol):
    def start(self) ->None:
        pass

    def stop(self) -> None:
        pass

    def sample(self) -> Iterator[Tuple[float,np.ndarray]]:
        pass