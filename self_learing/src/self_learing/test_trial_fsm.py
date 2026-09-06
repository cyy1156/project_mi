from __future__ import annotations

from self_learing.timing import TrialTiming
from self_learing.trail_fsm import  run_trail

def test_event_order() -> None:
    names:list[str]=[]

    def emit(n:str,t:float,**kw) -> None:
        names.append(n)

    timing:TrialTiming = TrialTiming(4,2,1,4,3)
    t_end=run_trail("Left",timing,emit)
    assert names[0]=="rest_start"
    assert "mi_start" in names
    assert names[-1]=="iti_start" or "iti" in names[-1]
    assert t_end ==4+2+4+3+1

def test_load_timing_from_config() -> None:
    from pathlib import Path

    from self_learing.timing import load_timing

    p=Path("config/protocol.yaml")
    if not p.is_file():
        return
    tm = load_timing(p)
    assert tm.imagine_s>0
