from __future__ import annotations

class LiveCapture:
    def __init__(self,source,buffer,bus)->None:
        self.source=source
        self.buffer=buffer
        self.bus=bus

    def run_n_sample(self,n:int)->None:
        for t,x in self.source.iter_samples(n):
            self.buffer.push(t,x)
            self.bus.publish(t,x)