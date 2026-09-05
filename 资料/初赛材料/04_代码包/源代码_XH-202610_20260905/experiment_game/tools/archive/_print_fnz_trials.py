import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
p = _REPO / "experiment_game" / "data" / "sessions" / "fnz_ws02_20260826_171537" / "v3_trial_features.jsonl"
if not p.is_file():
    print(f"skip: missing {p}", file=sys.stderr)
    raise SystemExit(0)

LABELS = {0: "Rest", 1: "Left", 2: "Right"}
for line in p.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    tid = r["trial_id"]
    if tid not in (11, 12, 13, 19, 20, 3, 6):
        continue
    print(tid, LABELS.get(r.get("label"), r.get("label")), r.get("verdict_text", "")[:80])
