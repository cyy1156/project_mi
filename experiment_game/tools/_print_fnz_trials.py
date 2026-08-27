import json
from pathlib import Path

p = Path(r"d:\MI\experiment_game\data\sessions\fnz_ws02_20260826_171537\v3_trial_features.jsonl")
LABELS = {0: "Rest", 1: "Left", 2: "Right"}
for line in p.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    tid = r["trial_id"]
    if tid not in (11, 12, 13, 19, 20, 3, 6):
        continue
    pj = r.get("primary_judge") or {}
    f = r.get("features") or {}
    g = f.get("grade", {}).get("grade", "-")
    print(f"T{tid:02d} label={LABELS[r['label']]} pred={LABELS.get(pj.get('pred'), '?')} grade={g}")
    if "mu_erd_contra" in f:
        print(f"  ERD_contra={f['mu_erd_contra']:.1f}% lat={f.get('laterality_pp'):.1f}pp")
        print(f"  checks={f.get('grade', {}).get('checks')}")
    if pj.get("p_three"):
        p3 = pj["p_three"]
        print(f"  p3 Rest/Left/Right = [{p3[0]:.3f}, {p3[1]:.3f}, {p3[2]:.3f}]")
    print()
