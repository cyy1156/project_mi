from __future__ import annotations

from typing import Any

from self_learing.src.self_learing.events import EventLogger
class MarkerSink:
    def __init__(self,logger:EventLogger)->None:
        self.logger=logger

    def emit(self,name:str,t:float,**files:Any)->None:
        self.logger.append(name,t,**files)