# -*- coding: utf-8 -*-
"""Rewrite cohort_map.py and summary_42.py as valid UTF-8 (ASCII + \\u escapes)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def w(name: str, text: str) -> None:
    (ROOT / name).write_text(text, encoding="utf-8", newline="\n")
    print("wrote", name, ROOT.joinpath(name).stat().st_size)

COHORT = r'''"""P0 identity merge -> analysis_42/cohort_map.json"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from paths import ANALYSIS, SUBJECTS

# explicit merge rules (plan section 0); others are singletons
MERGE_RULES: Dict[str, List[str]] = {
    "fnz": ["fnz", "xjh0828", "fnz0830", "fnz_1"],
    "cyy": ["cyy", "cyy0830"],
}

SKIP_DIRS = {
    "_analysis",
    "_backup_old_channel_order_20260829",
    "test",
    "learn_m00",
}

def _stem_person(sid: str) -> str:
    m = re.match(r"^([a-zA-Z]+)", sid)
    return (m.group(1) if m else sid).lower()

def build_cohort_map() -> Dict[str, Any]:
    present = sorted(
        d.name
        for d in SUBJECTS.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith("_")
    )
    assigned: set[str] = set()
    people: List[Dict[str, Any]] = []

    for person, members in MERGE_RULES.items():
        hit = [m for m in members if m in present]
        if not hit:
            continue
        assigned.update(hit)
        people.append(
            {
                "person_id": person,
                "member_ids": hit,
                "primary_id": sorted(hit, key=lambda x: (0 if re.search(r"\d{4}$", x) else 1, x))[
                    0
                ],
                "merge_rule": "explicit",
            }
        )

    for sid in present:
        if sid in assigned:
            continue
        # alias shell without date suffix when dated id exists
        if not re.search(r"\d{4}$", sid):
            stem = _stem_person(sid)
            if any(x.startswith(stem) and re.search(r"\d{4}$", x) for x in present):
                continue
        people.append(
            {
                "person_id": sid,
                "member_ids": [sid],
                "primary_id": sid,
                "merge_rule": "singleton",
            }
        )

    people.sort(key=lambda p: p["person_id"])
    return {
        "schema": "exp42_cohort_map_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_people": len(people),
        "n_member_dirs": sum(len(p["member_ids"]) for p in people),
        "people": people,
        "notes": "analysis unit = person_id",
    }

def main() -> Path:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    obj = build_cohort_map()
    path = ANALYSIS / "cohort_map.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[P0] wrote {path} n_people={obj['n_people']}")
    return path

if __name__ == "__main__":
    main()
'''

SUMMARY = r'''"""Write Day0 results into summary registry markdown."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from paths import ANALYSIS, SUMMARY_DIR

def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and (x != x)):
            return ""
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)
    except Exception:
        return ""

def write_registry() -> Path:
    cohort = json.loads((ANALYSIS / "cohort_map.json").read_text(encoding="utf-8"))
    arms = json.loads((ANALYSIS / "arms_day0.json").read_text(encoding="utf-8"))
    A, B, C, D, E = arms["A"], arms["B"], arms["C"], arms["D"], arms["E"]

    # Chinese headings via unicode escapes (encoding-safe source)
    t = {
        "title": "\u65b9\u6848\u56db\u5341\u4e8c \u00b7 \u771f\u4eba\u961f\u5217\u6df7\u6742\u5206\u89e3\u4e0e\u5749\u584c\u8bca\u65ad \u00b7 \u7ed3\u679c\u767b\u8bb0",
        "status": "Day0 \u5df2\u6267\u884c",
        "p0": "P0 \u00b7 \u8eab\u4efd\u5408\u5e76",
        "a": "A \u00b7 \u6df7\u6742\u5206\u89e3\uff08\u89c2\u6d4b\u4ee3\u7406\uff1b\u975e A1\u2212A2 \u5934\u5bf9\u5934\uff09",
        "b": "B \u00b7 \u75b2\u52b3\u4e0e\u8d28\u91cf\u5173\u8054",
        "c": "C \u00b7 \u4f1a\u8bdd\u7279\u5f81\u663e\u8457\u6027\u4e0e\u540c\u8d28\u6027",
        "d": "D \u00b7 \u5749\u584c\u4e09\u5c42\u5b9a\u4f4d",
        "e": "E \u00b7 BCI2a \u5bf9\u7167",
        "exec": "\u6267\u884c\u8bb0\u5f55",
    }

    lines = []
    lines.append(f"# {t['title']}")
    lines.append("")
    lines.append(
        f"> \u65b9\u6848\uff1a[../\u65b9\u6848.md](../\u65b9\u6848.md) \u00b7 \u72b6\u6001\uff1a**{t['status']}** \u00b7 "
        f"\u751f\u6210 `{datetime.now().isoformat(timespec='seconds')}`"
    )
    lines.append(
        "> \u5355\u4f4d\u7eaa\u5f8b\uff1a\u5206\u6790\u7528 person_id\uff1bA \u81c2\u5b8c\u6574 jackknife \u91cd\u8bad\u672a\u8dd1\uff0c\u73b0\u4e3a Leave-Next **\u89c2\u6d4b\u4ee3\u7406**\u3002"
    )
    lines.append("")
    lines.append(f"## {t['p0']}")
    lines.append("")
    lines.append("| person_id | member_ids | primary_id | rule |")
    lines.append("|---|---|---|---|")
    for p in cohort["people"]:
        lines.append(
            f"| {p['person_id']} | {' / '.join(p['member_ids'])} | {p['primary_id']} | {p['merge_rule']} |"
        )
    lines.append("")
    lines.append(f"n_people=**{cohort['n_people']}** (member_dirs={cohort['n_member_dirs']}).")
    lines.append("")
    lines.append(f"## {t['a']}")
    lines.append("")
    lines.append("| metric | mean | frac_pos | gate | verdict |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| R_vol_proxy=last-first MI | {_fmt(A['R_vol_proxy_mean'], 4)} | "
        f"{_fmt(A['frac_positive'], 3)} | >=+0.03 and >=2/3 | **{A['verdict']}** |"
    )
    lines.append(
        f"| R_recent_proxy=last-mid MI | {_fmt(A['R_recent_proxy_mean'], 4)} | - | aux | - |"
    )
    lines.append("| R_span / R_order / dA5 | - | - | needs A retrain | **pending GPU** |")
    lines.append("")
    lines.append(f"note: {A['note']}")
    lines.append("")
    lines.append(f"## {t['b']}")
    lines.append("")
    lines.append("| metric | result | read |")
    lines.append("|---|---|---|")
    lines.append(
        f"| frac Rest-mu slope>0 | {_fmt(B['frac_rest_mu_slope_pos'])} (n={B['n_people_slope']}) | {B['fatigue_verdict']} |"
    )
    lines.append(
        f"| frac |LI| slope<0 | {_fmt(B['frac_abs_li_slope_neg'])} | aux |"
    )
    sp = B["spearman_gain_vs_sat"]
    lines.append(
        f"| Spearman gain x sat | r={_fmt(sp['r'])}, p={_fmt(sp['p'])}, n={sp['n']} | "
        f"{'weak quality confound' if abs(sp.get('r') or 0) < 0.2 else 'check'} |"
    )
    sp2 = B["spearman_gain_vs_gap"]
    lines.append(
        f"| Spearman gain x gap | r={_fmt(sp2['r'])}, p={_fmt(sp2['p'])}, n={sp2['n']} | aux |"
    )
    sp3 = B["spearman_gain_vs_dprime"]
    lines.append(
        f"| Spearman gain x dprime | r={_fmt(sp3['r'])}, p={_fmt(sp3['p'])}, n={sp3['n']} | aux |"
    )
    wil = B["adjacent_dprime_delta"]
    lines.append(
        f"| adjacent dprime delta | median={_fmt(wil.get('median'))}, "
        f"p={_fmt(wil.get('p'))}, n={wil.get('n')} | drift proxy |"
    )
    lines.append("")
    lines.append(f"## {t['c']}")
    lines.append("")
    bins = C["dprime_bins"]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(
        f"| dprime bins strong/mid/weak | "
        f"{bins['strong_ge_1.0']} / {bins['mid_0.5_1.0']} / {bins['weak_lt_0.5']} "
        f"(weak_frac={_fmt(bins['weak_frac'])}) |"
    )
    lines.append(f"| median adjacent dprime gap | {_fmt(C['median_adjacent_dprime_gap'])} |")
    lines.append(f"| note | {C['note_friedman']} |")
    lines.append("")
    lines.append(f"## {t['d']}")
    lines.append("")
    lines.append("| label | n | frac |")
    lines.append("|---|---|---|")
    for k in ("signal", "optim", "readout", "mixed", "none"):
        lines.append(f"| {k} | {D['counts'][k]} | {_fmt(D['frac'][k])} |")
    lines.append("")
    lines.append("| person | dprime/AUC | max_class_frac | label |")
    lines.append("|---|---|---|---|")
    for p in D["per_person"]:
        lines.append(
            f"| {p['person_id']} | d={_fmt(p['med_dprime_lr'])}, "
            f"AUC={_fmt(p['med_probe_auc_lr'])} | {_fmt(p['max_class_frac_peak'])} | {p['label']} |"
        )
    lines.append("")
    lines.append(f"## {t['e']}")
    lines.append("")
    lines.append("| metric | sim | human |")
    lines.append("|---|---|---|")
    lines.append(
        f"| collapse frac | {E.get('sim_collapse_frac') or 'TBD'} | "
        f"{_fmt(E['human_collapse_frac'])} (n={E['human_n']}) |"
    )
    lines.append(f"| note | {E['sim_note']} | |")
    lines.append("")
    lines.append(f"## {t['exec']}")
    lines.append("")
    lines.append("```text")
    lines.append("cd code/train_lab/src/step/5070_exp42_confound_collapse_accpaper")
    lines.append("python run_day0.py --workers 4")
    lines.append("```")
    lines.append("")
    lines.append("| date | event |")
    lines.append("|------|------|")
    lines.append("| 2026-09-04 | registry skeleton |")
    lines.append(
        f"| {datetime.now().strftime('%Y-%m-%d')} | Day0 done: P0+feat+B/C/D/E + A proxy; A retrain pending |"
    )
    lines.append("")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    # filename: 结果登记表.md
    path = SUMMARY_DIR / "\u7ed3\u679c\u767b\u8bb0\u8868.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] wrote {path}")
    return path

if __name__ == "__main__":
    write_registry()
'''

w("cohort_map.py", COHORT)
w("summary_42.py", SUMMARY)
print("done")
