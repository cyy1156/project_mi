"""对齐索引：trial_table.csv + verify_report.json。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_events(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _eeg_span(path: Path) -> Optional[Tuple[float, float, int]]:
    if not path.is_file():
        return None
    times: List[float] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "lsl_time" not in reader.fieldnames:
            return None
        for row in reader:
            times.append(float(row["lsl_time"]))
    if not times:
        return None
    return times[0], times[-1], len(times)


def build_trial_table(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_trial: Dict[int, Dict[str, Any]] = defaultdict(dict)
    rejects = set()
    for ev in events:
        name = ev.get("event")
        tid = ev.get("trial_id")
        if name == "trial_reject" and tid is not None:
            rejects.add(int(tid))
        if tid is None:
            continue
        tid = int(tid)
        row = by_trial[tid]
        row["trial_id"] = tid
        if "phase" not in row and ev.get("phase"):
            row["phase"] = ev.get("phase")
        if ev.get("object"):
            row["object"] = ev["object"]
        if ev.get("scene"):
            row["scene"] = ev["scene"]
        if ev.get("learn_step") is not None:
            row["learn_step"] = ev.get("learn_step")
        if name in (
            "trial_start",
            "trial_end",
            "fixation",
            "cue",
            "mi_start",
            "mi_end",
            "rest_start",
            "rest_end",
            "transition",
        ):
            row[f"t_{name}"] = float(ev["t_lsl"])
        if name in ("trial_start", "cue", "mi_start") and ev.get("label") is not None:
            row["label"] = ev.get("label")

    rows = []
    for tid in sorted(by_trial.keys()):
        r = by_trial[tid]
        r["rejected"] = 1 if tid in rejects else 0
        if "t_mi_start" in r and "t_mi_end" in r:
            r["mi_dur"] = float(r["t_mi_end"]) - float(r["t_mi_start"])
        if "t_rest_start" in r and "t_rest_end" in r:
            r["rest_dur"] = float(r["t_rest_end"]) - float(r["t_rest_start"])
        rows.append(r)
    return rows


def write_trial_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "trial_id",
        "phase",
        "learn_step",
        "label",
        "object",
        "scene",
        "rejected",
        "t_trial_start",
        "t_fixation",
        "t_cue",
        "t_mi_start",
        "t_mi_end",
        "t_rest_start",
        "t_rest_end",
        "t_transition",
        "t_trial_end",
        "mi_dur",
        "rest_dur",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def verify_alignment(
    events: List[Dict[str, Any]],
    trial_rows: List[Dict[str, Any]],
    eeg_path: Optional[Path],
    *,
    tol_s: float = 0.05,
    require_acq: bool = True,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "passed": True,
        "checks": [],
        "errors": [],
    }

    def add(ok: bool, name: str, detail: str = "") -> None:
        report["checks"].append({"ok": ok, "name": name, "detail": detail})
        if not ok:
            report["passed"] = False
            report["errors"].append(f"{name}: {detail}")

    # monotonic
    prev = None
    regressions = 0
    for ev in events:
        t = float(ev["t_lsl"])
        if prev is not None and t + 1e-9 < prev:
            regressions += 1
        prev = t
    add(regressions == 0, "t_lsl_monotonic", f"regressions={regressions}")

    acquire = [r for r in trial_rows if r.get("phase") == "acquire"]
    mi_bad = []
    rest_bad = []
    for r in acquire:
        if r.get("rejected"):
            continue
        md = r.get("mi_dur")
        rd = r.get("rest_dur")
        if md is None or abs(float(md) - 4.0) > tol_s:
            mi_bad.append((r.get("trial_id"), md))
        if rd is None or abs(float(rd) - 4.0) > tol_s:
            rest_bad.append((r.get("trial_id"), rd))
    add(
        len(mi_bad) == 0,
        "acquire_mi_4s",
        f"bad={mi_bad[:5]} n_acquire={len(acquire)}",
    )
    add(
        len(rest_bad) == 0,
        "acquire_rest_4s",
        f"bad={rest_bad[:5]} n_acquire={len(acquire)}",
    )

    span = _eeg_span(eeg_path) if eeg_path else None
    if require_acq:
        if span is None:
            add(False, "eeg_present", f"missing {eeg_path}")
        else:
            t0, t1, n = span
            add(True, "eeg_present", f"samples={n} span={t1 - t0:.2f}s")
            outside = 0
            for r in acquire:
                for key in ("t_cue", "t_mi_start", "t_rest_start"):
                    tv = r.get(key)
                    if tv is None:
                        continue
                    if not (t0 <= float(tv) <= t1):
                        outside += 1
            add(outside == 0, "markers_within_eeg", f"outside={outside}")
    else:
        add(True, "eeg_optional", "acq disabled")

    return report


def write_alignment_bundle(
    session_root: Path,
    *,
    acq_enabled: bool = True,
) -> Dict[str, Any]:
    session_root = Path(session_root)
    events_path = session_root / "events.jsonl"
    eeg_path = session_root / "eeg.csv"
    if not eeg_path.is_file():
        cont = session_root / "continuous" / "eeg.csv"
        if cont.is_file():
            eeg_path = cont

    events = _load_events(events_path)
    rows = build_trial_table(events)
    align_dir = session_root / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    write_trial_table(align_dir / "trial_table.csv", rows)

    dict_path = align_dir / "marker_dictionary.json"
    names = sorted({str(e.get("event")) for e in events if e.get("event")})
    dict_path.write_text(
        json.dumps({"events": names}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = verify_alignment(
        events,
        rows,
        eeg_path if acq_enabled else None,
        require_acq=acq_enabled,
    )
    (align_dir / "verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
