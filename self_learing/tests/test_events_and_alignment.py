from __future__ import annotations

from self_learing.alignment import verify_events
from self_learing.events import  EventLogger,read_events

def test_roundtrip(tmp_path)->None:
    path=tmp_path/"events.json"
    lg=EventLogger(path)
    lg.append("rest_start",0.0,label="left")
    lg.append("mi_end",6.0,label="left")
    lg.close()
    rows=read_events(path)
    assert [r["name"] for r in rows]==["rest_start","mi_end"]


def test_alignment_fail()->None:
    ev=[{"name":"x","t":999.0}]
    rep =verify_events(ev,0.0,10.0)
    assert rep["passed"] is False

def test_alignment_pass()->None:
    ev=[{"name":"x","t":1.0}]
    rep =verify_events(ev,0,10.0)
    assert rep["passed"] is True