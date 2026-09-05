"""P0 identity merge -> analysis_42/cohort_map.json"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from paths import ANALYSIS, SUBJECTS

# explicit merge rules (plan section 0); others are singletons
# 2026-09-05 口径修订：fnz0828 与 fnz0830 按登记行各自为独立个体（n_people=17，
# 与 v4 报告 §3.6 对齐）；同日重复壳 fnz / fnz_1 仍归入 fnz0828。
MERGE_RULES: Dict[str, List[str]] = {
    "fnz0828": ["fnz", "fnz0828", "fnz_1"],
    "cyy0830": ["cyy", "cyy0830"],
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
