import json
from pathlib import Path
root = Path("D:/MI/experiment_game/data/subjects")
by_sid = {}
for p in root.glob("*/models/ft_runs/*leave_next*all4*f5_summary.json"):
    d = json.loads(p.read_text(encoding="utf-8"))
    sid = str(d.get("subject_id") or p.parent.parent.parent.name)
    sid = {"fnz0828": "xjh0828"}.get(sid, sid)
    stamp = p.name.split("_")[0]
    prev = by_sid.get(sid)
    if prev is None or stamp >= prev["stamp"]:
        by_sid[sid] = {"stamp": stamp, "path": p.name}
print("has xjh0828", "xjh0828" in by_sid, by_sid.get("xjh0828"))
print("has fnz0828", "fnz0828" in by_sid)
print("n", len(by_sid))
print("dirs", sorted(p.name for p in root.iterdir() if p.is_dir() and ("xjh" in p.name or p.name.startswith("fnz"))))
print("old gone", not (root / "fnz0828").exists())
