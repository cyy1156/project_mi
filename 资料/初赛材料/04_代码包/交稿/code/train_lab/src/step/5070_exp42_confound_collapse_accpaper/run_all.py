# -*- coding: utf-8 -*-
"""Exp42 one-shot runner: Day0 (P0/B/C/D/E) + Day1 (Arm A jackknife) + summary.

Usage (from this directory, conda env with torch/sklearn):
  python run_all.py --stage all
  python run_all.py --stage day0 --workers 4
  python run_all.py --stage day1 --seeds 0,1,2,3,4
  python run_all.py --stage day1 --dry-run
  python run_all.py --stage day1 --people syj0828,fnz --max-jobs 2
  python run_all.py --stage summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

# Resolve real repo root (handles D:\MI <-> D:\code junctions)
from paths import REPO as _REPO  # noqa: E402

for _p in (_REPO, _REPO / "code"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stage_day0(*, workers: int, skip_extract: bool) -> Dict[str, Any]:
    from cohort_map import build_cohort_map, main as p0_main
    from extract_features import run_extract
    from parse_leave_next import run_parse
    from arms_day0 import run_arms
    from arm_E_exp32 import run_arm_E
    from summary_42 import write_registry

    t0 = time.time()
    cohort_path = p0_main()
    cohort = build_cohort_map()
    parse_path = run_parse(cohort)
    extract_path = None
    if not skip_extract:
        extract_path = run_extract(cohort, max_workers=workers)
    arms_path = run_arms()
    e_path = run_arm_E()
    # merge E into arms_day0.json for registry
    from paths import ANALYSIS

    arms_file = ANALYSIS / "arms_day0.json"
    if arms_file.is_file():
        arms = json.loads(arms_file.read_text(encoding="utf-8"))
        arms["E"] = {
            "arm": "E",
            "human_collapse_frac": e_path.get("human_collapse_frac"),
            "human_n": (e_path.get("human") or {}).get("n"),
            "sim_collapse_frac": e_path.get("sim_collapse_frac"),
            "sim_note": (e_path.get("sim_subjects") or {}).get("note")
            or (e_path.get("exp32_docs") or {}).get("note"),
            "delta_human_minus_sim": e_path.get("delta_human_minus_sim"),
        }
        arms_file.write_text(json.dumps(arms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reg = write_registry()
    return {
        "stage": "day0",
        "ok": True,
        "seconds": round(time.time() - t0, 1),
        "cohort": str(cohort_path),
        "parse": str(parse_path),
        "extract": str(extract_path) if extract_path else "skipped",
        "arms": str(arms_path),
        "registry": str(reg),
    }


def stage_day1(
    *,
    people: Optional[List[str]],
    arms: List[str],
    seeds: List[int],
    device: Optional[str],
    dry_run: bool,
    max_jobs: Optional[int],
) -> Dict[str, Any]:
    from arm_A_jackknife import run_arm_A

    t0 = time.time()
    summary = run_arm_A(
        people=people,
        arms=arms,
        seeds=seeds,
        device=device,
        dry_run=dry_run,
        max_jobs=max_jobs,
    )
    return {
        "stage": "day1",
        "ok": True,
        "seconds": round(time.time() - t0, 1),
        "summary": summary,
    }


def stage_summary() -> Dict[str, Any]:
    from paths import ANALYSIS, OUT_ROOT
    from summary_42 import write_registry
    from arm_E_exp32 import run_arm_E

    e = run_arm_E()
    arms_file = ANALYSIS / "arms_day0.json"
    if arms_file.is_file():
        arms = json.loads(arms_file.read_text(encoding="utf-8"))
    else:
        arms = {
            "schema": "exp42_arms_day0_v1",
            "generated_at": _ts(),
            "A": {"verdict": "pending_day0", "note": "day0 not run", "R_vol_proxy_mean": None,
                  "frac_positive": None, "R_recent_proxy_mean": None},
            "B": {"fatigue_verdict": "pending_day0", "n_people_slope": 0,
                  "frac_rest_mu_slope_pos": None, "frac_abs_li_slope_neg": None,
                  "spearman_gain_vs_sat": {"r": None, "p": None, "n": 0},
                  "spearman_gain_vs_gap": {"r": None, "p": None, "n": 0},
                  "spearman_gain_vs_dprime": {"r": None, "p": None, "n": 0},
                  "adjacent_dprime_delta": {"median": None, "p": None, "n": 0}},
            "C": {"dprime_bins": {"strong_ge_1.0": 0, "mid_0.5_1.0": 0, "weak_lt_0.5": 0, "weak_frac": None},
                  "median_adjacent_dprime_gap": None, "note_friedman": "pending_day0"},
            "D": {"counts": {"signal": 0, "optim": 0, "readout": 0, "mixed": 0, "none": 0},
                  "frac": {"signal": 0, "optim": 0, "readout": 0, "mixed": 0, "none": 0},
                  "per_person": []},
            "E": {},
            "partial": True,
        }
    arms["E"] = {
        "arm": "E",
        "human_collapse_frac": e.get("human_collapse_frac"),
        "human_n": (e.get("human") or {}).get("n"),
        "sim_collapse_frac": e.get("sim_collapse_frac"),
        "sim_note": (e.get("sim_subjects") or {}).get("note")
        or (e.get("exp32_docs") or {}).get("note"),
        "delta_human_minus_sim": e.get("delta_human_minus_sim"),
    }
    a_agg = ANALYSIS / "arm_A_aggregate.json"
    if a_agg.is_file():
        A = json.loads(a_agg.read_text(encoding="utf-8"))
        arms["A_full"] = A
        if A.get("verdict"):
            arms.setdefault("A", {})["full_verdict"] = A.get("verdict")
            arms["A"]["R_vol_mean_full"] = A.get("R_vol_mean")
            arms["A"]["frac_positive_full"] = A.get("frac_positive_R_vol")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    arms_file.write_text(json.dumps(arms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "replay_42_summary.json").write_text(
        json.dumps(arms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reg = write_registry()
    return {"stage": "summary", "ok": True, "registry": str(reg)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp42 run-all orchestrator")
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=("all", "day0", "day1", "a", "summary", "e"),
        help="all=day0+day1+summary; a=day1 only",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--people", type=str, default="", help="comma ids for Arm A")
    ap.add_argument("--arms", type=str, default="A1,A2,A3,A4,A5")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--dry-run", action="store_true", help="Arm A: queue only, no FT")
    ap.add_argument("--max-jobs", type=int, default=0, help="Arm A smoke limit; 0=all")
    ap.add_argument(
        "--day0-only-if-missing",
        action="store_true",
        help="skip day0 when analysis_42/arms_day0.json already exists",
    )
    args = ap.parse_args()

    people = [x.strip() for x in args.people.split(",") if x.strip()] or None
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = args.device or None
    max_jobs = args.max_jobs or None

    from paths import ANALYSIS, OUT_ROOT

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log: Dict[str, Any] = {
        "schema": "exp42_run_all_log_v1",
        "started_at": _ts(),
        "stage_arg": args.stage,
        "results": [],
    }

    stage = args.stage
    if stage == "a":
        stage = "day1"

    try:
        if stage in ("all", "day0"):
            arms0 = ANALYSIS / "arms_day0.json"
            if args.day0_only_if_missing and arms0.is_file() and stage == "all":
                print("[run_all] skip day0 (arms_day0.json exists)")
                log["results"].append({"stage": "day0", "ok": True, "skipped": True})
            else:
                print("\n===== Exp42 Day0: P0 + features + B/C/D + E =====")
                log["results"].append(
                    stage_day0(workers=args.workers, skip_extract=args.skip_extract)
                )

        if stage in ("all", "day1"):
            print("\n===== Exp42 Day1: Arm A jackknife (GPU) =====")
            log["results"].append(
                stage_day1(
                    people=people,
                    arms=arms,
                    seeds=seeds,
                    device=device,
                    dry_run=args.dry_run,
                    max_jobs=max_jobs,
                )
            )

        if stage == "e":
            from arm_E_exp32 import run_arm_E

            log["results"].append({"stage": "e", "ok": True, "data": run_arm_E()})

        if stage in ("all", "summary", "day1", "day0", "e"):
            # always refresh registry after work stages
            if stage != "summary" or True:
                print("\n===== Exp42 summary registry =====")
                log["results"].append(stage_summary())

        log["finished_at"] = _ts()
        log["ok"] = all(r.get("ok", True) for r in log["results"])
    except Exception as exc:
        log["finished_at"] = _ts()
        log["ok"] = False
        log["error"] = str(exc)
        raise
    finally:
        path = OUT_ROOT / "run_all_log.json"
        path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n[run_all] log -> {path} ok={log.get('ok')}")


if __name__ == "__main__":
    main()
