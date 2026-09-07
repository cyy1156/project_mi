"""?? Leave-Next summary + ft_runs release_gate -> ??????????"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paths import ANALYSIS, SUBJECTS


def _prefer_summary(sid: str) -> Optional[Path]:
    ft = SUBJECTS / sid / "models" / "ft_runs"
    if not ft.is_dir():
        return None
    all4 = sorted(ft.glob("*leave_next*all4*f5_summary.json"))
    if all4:
        return all4[-1]
    any_ = sorted(ft.glob("*leave_next*f5_summary.json"))
    return any_[-1] if any_ else None


def _rows_of(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d, {}
    meta = {k: d.get(k) for k in ("ft_scope", "stamp", "subject_id", "protocol") if k in d}
    for k in ("stages", "rows", "summary", "results", "ramp"):
        if isinstance(d.get(k), list):
            return d[k], meta
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "r_stage" in v[0]:
            return v, meta
    return [], meta


def _mi_of(row: Dict[str, Any]) -> float:
    f5 = row.get("f5_ft") or row.get("f5") or {}
    if isinstance(f5, dict) and f5.get("mi_acc") is not None:
        return float(f5["mi_acc"])
    for k in ("mi", "mi_acc", "f5_mi"):
        if row.get(k) is not None:
            return float(row[k])
    return float("nan")


def _gate_of(row: Dict[str, Any]) -> Optional[bool]:
    if "release_pass" in row:
        return bool(row["release_pass"])
    if "pass" in row:
        return bool(row["pass"])
    return None


def parse_member_leave_next(member_id: str) -> Dict[str, Any]:
    path = _prefer_summary(member_id)
    if path is None:
        return {"member_id": member_id, "error": "no_summary"}
    rows, meta = _rows_of(path)
    stages = []
    for r in rows:
        stages.append(
            {
                "r": int(r.get("r_stage") or r.get("r") or len(stages) + 1),
                "train": r.get("train"),
                "heldout": r.get("heldout"),
                "use_replay": r.get("use_replay"),
                "mi": _mi_of(r),
                "win_smooth": float(r.get("heldout_acc_smooth") or r.get("heldout_acc") or float("nan")),
                "win_raw": float(r.get("heldout_acc_raw") or float("nan")),
                "max_class_frac": float(r.get("max_class_frac") or float("nan")),
                "pass": _gate_of(r),
                "out_dir": r.get("out_dir"),
            }
        )
    gates = []
    ft = SUBJECTS / member_id / "models" / "ft_runs"
    if ft.is_dir():
        for d in sorted(ft.iterdir()):
            if not d.is_dir() or "leave_next" not in d.name:
                continue
            gpath = d / "release_gate.json"
            if not gpath.is_file():
                continue
            try:
                g = json.loads(gpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            gates.append(
                {
                    "run": d.name,
                    "pass": bool(g.get("pass")),
                    "heldout_acc": float(g.get("heldout_acc") or float("nan")),
                    "max_class_frac": float(g.get("max_class_frac") or float("nan")),
                    "pred_labels": (
                        g.get("pred_labels")
                        or g.get("fusion", {}).get("heldout_pred_dist", {}).get("pred_counts")
                    ),
                }
            )
    first_mi = stages[0]["mi"] if stages else float("nan")
    last_mi = stages[-1]["mi"] if stages else float("nan")
    return {
        "member_id": member_id,
        "summary": path.name,
        "meta": meta,
        "n_rounds": len(stages),
        "stages": stages,
        "first_mi": first_mi,
        "last_mi": last_mi,
        "delta_mi": (last_mi - first_mi) if stages else float("nan"),
        "last_pass": stages[-1]["pass"] if stages else None,
        "max_class_frac_peak": max((s["max_class_frac"] for s in stages), default=float("nan")),
        "gates_all_leave_next": gates,
        "collapse_any": any((s.get("max_class_frac") or 0) >= 0.85 for s in stages)
        or any((g.get("max_class_frac") or 0) >= 0.85 for g in gates),
    }


def run_parse(cohort: Dict[str, Any]) -> Path:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows = []
    for person in cohort["people"]:
        member_rows = []
        for mid in person["member_ids"]:
            member_rows.append(parse_member_leave_next(mid))
        prim = person["primary_id"]
        prim_row = next((m for m in member_rows if m["member_id"] == prim), member_rows[0])
        rows.append(
            {
                "person_id": person["person_id"],
                "primary_id": prim,
                "members": member_rows,
                "primary": prim_row,
            }
        )
    out = {
        "schema": "exp42_leave_next_parse_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_people": len(rows),
        "rows": rows,
    }
    path = ANALYSIS / "leave_next_parse.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[parse] wrote {path}")
    return path


if __name__ == "__main__":
    from cohort_map import build_cohort_map

    run_parse(build_cohort_map())
