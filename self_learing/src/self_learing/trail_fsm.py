from __future__ import annotations

from self_learing.timing import TrialTiming

from typing import Callable

EmitFn =Callable[...,None]#emit(name,t,**fields)

def run_trail(label:str,timing:TrialTiming,emit,*,t0:float=0.0)->float:
    t=float(t0)

    emit("rest_start",t,label=label)
    t +=timing.rest_s
    emit("rest_end",t,label=label)

    emit("pre_start",t,label=label)
    t+=timing.prep_s
    emit("cue_s",t,label=label)
    t+=timing.cue_s
    emit("cue_end",t,label=label)
    emit("mi_start",t,label=label)
    t+=timing.imagine_s
    emit("mi_end",t,label=label)

    emit("iti_start",t,label=label)
    t+=timing.iti_s
    return t


