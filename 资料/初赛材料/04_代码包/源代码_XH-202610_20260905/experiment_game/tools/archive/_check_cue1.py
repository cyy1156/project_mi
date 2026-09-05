from pathlib import Path
import json
import re

for p in [
    "experiment_game/config/operator_defaults.json",
    "experiment_game/config/operator_defaults.example.json",
]:
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    vo = d["experiment"]["v3_overrides"]
    assert vo["cue_s"] == 1, (p, vo["cue_s"])
    assert vo["block_gap_s"] == 30, (p, vo["block_gap_s"])
    print("OK", p, "cue", vo["cue_s"], "gap", vo["block_gap_s"])

h = Path("experiment_game/web/operator.html").read_text(encoding="utf-8")
m = re.search(r'name="v3_cue_s"[^>]*value="(\d+)"', h)
assert m and m.group(1) == "1", m
m2 = re.search(r'name="v3_block_gap_s"[^>]*value="(\d+)"', h)
assert m2 and m2.group(1) == "30", m2
assert "Align 2/1/4/3+Rest4" in h
js = Path("experiment_game/web/js/operator.js").read_text(encoding="utf-8")
assert "cue_s: 1" in js
assert "operator_defaults_v2" in js
print("OK html cue", m.group(1), "gap", m2.group(1), "preset+storage bumped")
print("ALL OK")
