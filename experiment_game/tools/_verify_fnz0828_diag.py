#!/usr/bin/env python3
"""核实 fnz0828_问题诊断报告.md 关键数据主张。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
ROOT = _REPO / "experiment_game" / "data" / "subjects"


def eeg_stats(sess: Path):
    p = sess / "eeg.csv"
    if not p.is_file():
        return None
    with p.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        rows = 0
        t0 = t1 = None
        for row in r:
            rows += 1
            try:
                t = float(row[0])
            except Exception:
                continue
            if t0 is None:
                t0 = t
            t1 = t
    return rows, (None if t0 is None else (t1 - t0))


def events_span(sess: Path):
    p = sess / "events.jsonl"
    if not p.is_file():
        return None, 0, set()
    ts = []
    types = set()
    n = 0
    for line in p.open(encoding="utf-8"):
        n += 1
        try:
            o = json.loads(line)
        except Exception:
            continue
        types.add(str(o.get("type") or o.get("event") or ""))
        t = o.get("t") or o.get("lsl_time") or o.get("time")
        if t is not None:
            ts.append(float(t))
    span = (max(ts) - min(ts)) if ts else None
    return span, n, types


def main() -> None:
    for sub in ("syj0828", "fnz0828"):
        print("====", sub)
        base = ROOT / sub / "sessions"
        for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
            er = eeg_stats(d)
            es, _en, types = events_span(d)
            sid = "?"
            meta = d / "session.meta.json"
            if meta.is_file():
                sid = json.loads(meta.read_text(encoding="utf-8")).get("session_id", "?")
            erows = "—" if not er else str(er[0])
            espan = "—" if not er or er[1] is None else f"{er[1]:.1f}"
            evsp = "—" if es is None else f"{es:.1f}"
            print(
                f"  {sid:4} rows={erows:>7} eeg_s={espan:>7} ev_s={evsp:>7} "
                f"eeg.meta={(d / 'eeg.meta.json').is_file()} "
                f"session_end={'session_end' in types}"
            )
            if sid == "ws07":
                print("    event_types sample:", sorted(t for t in types if t)[:20])

    print("\n==== FT reports (heldout / gate)")
    for sub in ("syj0828", "fnz0828"):
        ft_root = ROOT / sub / "models" / "ft_runs"
        if not ft_root.is_dir():
            continue
        for d in sorted(ft_root.iterdir()):
            meta = d / "meta.json"
            gate = d / "release_gate.json"
            if not meta.is_file():
                continue
            m = json.loads(meta.read_text(encoding="utf-8"))
            g = json.loads(gate.read_text(encoding="utf-8")) if gate.is_file() else {}
            three = (m.get("three") or {})
            pred = ((three.get("heldout_pred_dist") or {}).get("pred_counts") or {})
            has_pt = (d / "best_task.pt").is_file() and (d / "best_three.pt").is_file()
            print(
                f"  {sub} {d.name}: three "
                f"{three.get('acc_before_heldout')}→{three.get('acc_after_heldout')} "
                f"task→{ (m.get('task') or {}).get('acc_after_heldout')} "
                f"pred={pred} gate={g.get('pass')} pt={has_pt} "
                f"train={m.get('train_sessions') or m.get('source_run')} "
                f"hold={m.get('heldout_sessions')}"
            )


if __name__ == "__main__":
    main()
