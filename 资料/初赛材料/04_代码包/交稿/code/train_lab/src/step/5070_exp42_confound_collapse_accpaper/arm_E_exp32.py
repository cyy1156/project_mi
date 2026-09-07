# -*- coding: utf-8 -*-
"""Exp42 Arm E: BCI2a / Exp32 control anchor vs human collapse rate."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from paths import ANALYSIS, EXP41_SUMMARY, OUT_ROOT, REPO


def _find_exp32_docs() -> List[Path]:
    hits = sorted(REPO.rglob("32_*accpaper"))
    return [p for p in hits if p.is_dir()]


def _parse_max_class_from_text(text: str) -> List[float]:
    vals = []
    for m in re.finditer(r"max_class_frac\s*[=:]\s*([0-9]*\.?[0-9]+)", text, flags=re.I):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            continue
    return vals


def _scan_sim_subjects() -> Dict[str, Any]:
    sim_root = REPO / "experiment_game" / "data" / "sim_subjects"
    if not sim_root.is_dir():
        return {"n": 0, "collapse_frac": None, "note": "no sim_subjects dir"}
    collapses = 0
    n = 0
    per = []
    for d in sorted(sim_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        ft = d / "models" / "ft_runs"
        if not ft.is_dir():
            continue
        peaks = []
        for gpath in ft.rglob("release_gate.json"):
            try:
                g = json.loads(gpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            v = g.get("max_class_frac")
            if v is not None:
                peaks.append(float(v))
        if not peaks:
            continue
        n += 1
        peak = max(peaks)
        coll = peak >= 0.85
        collapses += int(coll)
        per.append({"id": d.name, "max_class_frac_peak": peak, "collapse": coll})
    return {
        "n": n,
        "collapse_frac": (collapses / n) if n else None,
        "n_collapse": collapses,
        "per": per,
        "note": "scanned sim_subjects release_gate.json",
    }


def _scan_exp32_tables() -> Dict[str, Any]:
    docs = _find_exp32_docs()
    vals: List[float] = []
    files: List[str] = []
    for root in docs:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".md", ".json", ".txt", ".csv"):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            found = _parse_max_class_from_text(text)
            if found:
                vals.extend(found)
                files.append(str(p.relative_to(REPO)))
    if not vals:
        return {"n_values": 0, "collapse_frac": None, "files": files, "note": "no max_class_frac in Exp32 docs"}
    n_hi = sum(1 for v in vals if v >= 0.85)
    return {
        "n_values": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "collapse_frac_values": n_hi / len(vals),
        "files": files[:20],
        "note": "parsed max_class_frac mentions from Exp32 artifacts",
    }


def _human_from_leave_next() -> Dict[str, Any]:
    path = ANALYSIS / "leave_next_parse.json"
    if not path.is_file():
        return {"n": 0, "collapse_frac": None, "note": "run Day0 parse first"}
    ln = json.loads(path.read_text(encoding="utf-8"))
    rows = ln.get("rows") or []
    n = len(rows)
    coll = sum(1 for r in rows if (r.get("primary") or {}).get("collapse_any"))
    return {
        "n": n,
        "n_collapse": coll,
        "collapse_frac": coll / n if n else None,
        "note": "from leave_next_parse.json",
    }


def run_arm_E() -> Dict[str, Any]:
    human = _human_from_leave_next()
    sim = _scan_sim_subjects()
    exp32 = _scan_exp32_tables()
    sim_frac = sim.get("collapse_frac")
    if sim_frac is None:
        sim_frac = exp32.get("collapse_frac_values")
    out = {
        "schema": "exp42_arm_E_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "human": human,
        "sim_subjects": sim,
        "exp32_docs": exp32,
        "sim_collapse_frac": sim_frac,
        "human_collapse_frac": human.get("collapse_frac"),
        "delta_human_minus_sim": (
            float(human["collapse_frac"]) - float(sim_frac)
            if human.get("collapse_frac") is not None and sim_frac is not None
            else None
        ),
        "read": (
            "if sim also high max_class_frac -> recipe/optim layer; "
            "if only human high -> person/state factors"
        ),
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS / "arm_E.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_ROOT / "arm_E.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[E] human_collapse={out['human_collapse_frac']} "
        f"sim_collapse={out['sim_collapse_frac']} -> {path}"
    )
    return out


if __name__ == "__main__":
    run_arm_E()
