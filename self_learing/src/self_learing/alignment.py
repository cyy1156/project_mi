from __future__ import annotations

from typing import Any,Dict,List

def verify_events(events: List[Dict[str,Any]],t0:float,t1:float) -> Dict[str,Any]:
    checks=[]
    ok=True
    for e in events:
        t =float(e["t"])
        in_range=(t0<=t<=t1)
        checks.append({"name":e["name"],"t":t,"in_range":in_range})
        ok=ok and in_range
        return {"passed":ok,"checks":checks}