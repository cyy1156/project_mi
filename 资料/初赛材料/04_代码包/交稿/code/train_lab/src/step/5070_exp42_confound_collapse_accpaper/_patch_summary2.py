# -*- coding: utf-8 -*-
"""Patch summary_42.write_registry to tolerate missing day0 arms."""
from pathlib import Path

p = Path(__file__).resolve().parent / "summary_42.py"
t = p.read_text(encoding="utf-8")
old = '''def write_registry() -> Path:
    cohort = json.loads((ANALYSIS / "cohort_map.json").read_text(encoding="utf-8"))
    arms = json.loads((ANALYSIS / "arms_day0.json").read_text(encoding="utf-8"))
    A, B, C, D, E = arms["A"], arms["B"], arms["C"], arms["D"], arms["E"]
'''
new = '''def write_registry() -> Path:
    cohort_path = ANALYSIS / "cohort_map.json"
    if not cohort_path.is_file():
        raise FileNotFoundError(f"missing {cohort_path}; run day0/P0 first")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    arms_path = ANALYSIS / "arms_day0.json"
    if arms_path.is_file():
        arms = json.loads(arms_path.read_text(encoding="utf-8"))
    else:
        empty = {"note": "day0 not run yet"}
        arms = {
            "A": {"R_vol_proxy_mean": float("nan"), "frac_positive": float("nan"),
                  "R_recent_proxy_mean": float("nan"), "verdict": "pending_day0", "note": empty["note"]},
            "B": {"frac_rest_mu_slope_pos": float("nan"), "n_people_slope": 0,
                  "fatigue_verdict": "pending_day0", "frac_abs_li_slope_neg": float("nan"),
                  "spearman_gain_vs_sat": {"r": float("nan"), "p": float("nan"), "n": 0},
                  "spearman_gain_vs_gap": {"r": float("nan"), "p": float("nan"), "n": 0},
                  "spearman_gain_vs_dprime": {"r": float("nan"), "p": float("nan"), "n": 0},
                  "adjacent_dprime_delta": {"median": float("nan"), "p": float("nan"), "n": 0}},
            "C": {"dprime_bins": {"strong_ge_1.0": 0, "mid_0.5_1.0": 0, "weak_lt_0.5": 0, "weak_frac": float("nan")},
                  "median_adjacent_dprime_gap": float("nan"), "note_friedman": "pending_day0"},
            "D": {"counts": {"signal": 0, "optim": 0, "readout": 0, "mixed": 0, "none": 0},
                  "frac": {"signal": 0, "optim": 0, "readout": 0, "mixed": 0, "none": 0},
                  "per_person": []},
            "E": {"sim_collapse_frac": None, "human_collapse_frac": None, "human_n": 0, "sim_note": "pending_day0"},
        }
        a_agg = ANALYSIS / "arm_A_aggregate.json"
        if a_agg.is_file():
            arms["A_full"] = json.loads(a_agg.read_text(encoding="utf-8"))
    A, B, C, D, E = arms["A"], arms["B"], arms["C"], arms["D"], arms["E"]
'''
if old not in t:
    raise SystemExit("anchor not found")
p.write_text(t.replace(old, new, 1), encoding="utf-8")
print("summary_42 patched")
