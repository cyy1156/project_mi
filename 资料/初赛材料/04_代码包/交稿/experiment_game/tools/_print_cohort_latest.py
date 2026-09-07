#!/usr/bin/env python3
"""Print latest Leave-Next F5 summary per cohort subject."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "subjects"
SUBJECTS = ["syj0828", "xjh0828", "cyy0830", "fnz0830", "wzr0830", "xj0830"]

def latest_summary(sid: str) -> Path | None:
    ft = ROOT / sid / "models" / "ft_runs"
    cands = sorted(
        ft.glob(f"*{sid}*leave_next*f5_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None

def f5_str(pack: dict | None) -> str:
    if not pack:
        return "—"
    bl = pack.get("by_label") or {}
    left, right, rest = bl.get("Left", {}), bl.get("Right", {}), bl.get("Rest", {})
    mi_ok = int(left.get("ok", 0)) + int(right.get("ok", 0))
    mi_n = int(left.get("n", 0)) + int(right.get("n", 0))
    r_ok, r_n = int(rest.get("ok", 0)), int(rest.get("n", 0))
    sc, sm = pack.get("score"), pack.get("score_max")
    scs = f"{sc:.1f}/{sm:.1f}" if sc is not None and sm else "—"
    return f"MI {mi_ok}/{mi_n} · Rest {r_ok}/{r_n} · {scs}"

def sess_key(s: str) -> str:
    for p in ("ws", "w"):
        i = s.find(p)
        if i >= 0:
            j = i + 2
            while j < len(s) and s[j].isdigit():
                j += 1
            return s[i:j]
    return "?"

def train_eval(row: dict) -> str:
    tr = row.get("train") or []
    keys = [sess_key(str(t)) for t in tr] if isinstance(tr, list) else ["?"]
    hold = sess_key(str(row.get("heldout") or ""))
    return f"{'+'.join(keys)}→{hold}"

def main() -> None:
    for sid in SUBJECTS:
        sp = latest_summary(sid)
        print("=" * 70)
        print(sid)
        if sp is None:
            print("  (无 summary)")
            continue
        payload = json.loads(sp.read_text(encoding="utf-8"))
        stamp = sp.name.split(f"_{sid}")[0]
        print(f"  stamp: {stamp}")
        rows = payload.get("rows") or []
        print(f"  档数: {len(rows)}")
        for row in rows:
            r = row.get("r_stage")
            sm = row.get("heldout_acc") or row.get("heldout_acc_smooth")
            raw = row.get("heldout_acc_raw", sm)
            gate = "PASS" if row.get("release_pass") else "FAIL"
            rep = "on" if row.get("use_replay") else "off"
            pred = row.get("pred_labels") or {}
            pred_s = ", ".join(f"{k}:{v}" for k, v in sorted(pred.items()))
            print(
                f"  R{r} {train_eval(row):<24} rep={rep:<3} "
                f"smooth={sm:.3f} raw={raw:.3f} {gate}  "
                f"F5={f5_str(row.get('f5_ft'))}"
            )
            print(f"      pred: {pred_s}")
        if rows:
            last = rows[-1]
            sm = last.get("heldout_acc") or last.get("heldout_acc_smooth")
            raw = last.get("heldout_acc_raw", sm)
            print(
                f"  >> 末档: {train_eval(last)} | smooth={sm:.3f} raw={raw:.3f} "
                f"gate={'PASS' if last.get('release_pass') else 'FAIL'} | "
                f"F5 FT {f5_str(last.get('f5_ft'))}"
            )

if __name__ == "__main__":
    main()
