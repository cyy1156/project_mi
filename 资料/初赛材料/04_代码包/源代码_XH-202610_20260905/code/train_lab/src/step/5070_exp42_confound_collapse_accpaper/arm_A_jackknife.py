# -*- coding: utf-8 -*-
"""Exp42 Arm A jackknife: A1-A5 train-set ablations on last Leave-Next heldout.

Outputs under code/train_lab/out/5070_exp42_confound_collapse/A/
Never writes to experiment_game/.../models/current.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from paths import ANALYSIS, OUT_ROOT, REPO, SUBJECTS  # noqa: E402

_REPO = REPO
for _p in (_REPO, _REPO / "code"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

A_ROOT = OUT_ROOT / "A"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ARMS = ("A1", "A2", "A3", "A4", "A5")
A5_REPLAY = (0.05, 0.10, 0.20)


def _as_float(x: Any, default: float = float("nan")) -> float:
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _mi_from_f5_blob(f5_blob: Any) -> float:
    """eval_f5_e1f returns {window_acc, f5:{mi_acc,...}}; tolerate flat mi_acc too."""
    if not isinstance(f5_blob, dict):
        return _as_float(f5_blob)
    nested = f5_blob.get("f5") if isinstance(f5_blob.get("f5"), dict) else {}
    return _as_float(nested.get("mi_acc", f5_blob.get("mi_acc")))


def _ensure_repo_path() -> None:
    for p in (_REPO, _REPO / "code", _PKG):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _import_leave_next():
    _ensure_repo_path()
    try:
        from experiment_game.tools.run_leave_next_e1f_task_ramp import (
            _list_v3_sessions,
            _ramp_for_subject,
            _session_dirs,
            eval_f5_e1f,
        )
        from adapt_engine.e1f import E1fRegistry, E1fStackConfig
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{exc}; sys.path[0:5]={sys.path[:5]!r}; _REPO={_REPO}"
        ) from exc

    return {
        "list_v3": _list_v3_sessions,
        "ramp": _ramp_for_subject,
        "session_dirs": _session_dirs,
        "eval_f5_e1f": eval_f5_e1f,
        "E1fRegistry": E1fRegistry,
        "E1fStackConfig": E1fStackConfig,
    }


def _load_cohort() -> Dict[str, Any]:
    path = ANALYSIS / "cohort_map.json"
    if not path.is_file():
        from cohort_map import build_cohort_map, main as p0_main

        p0_main()
    return json.loads(path.read_text(encoding="utf-8"))


def _last_stage(member_id: str, helpers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        by_ws = helpers["list_v3"](member_id)
        ramp = helpers["ramp"](member_id, by_ws)
    except Exception as exc:
        return {"member_id": member_id, "error": f"ramp_fail:{exc}"}
    if not ramp:
        return {"member_id": member_id, "error": "empty_ramp"}
    train_keys, hold_key, use_replay = ramp[-1]
    train_dirs: List[Path] = []
    for k in train_keys:
        train_dirs.extend(helpers["session_dirs"](by_ws, k))
    hold_dirs = helpers["session_dirs"](by_ws, hold_key)
    return {
        "member_id": member_id,
        "by_ws_keys": sorted(by_ws.keys()),
        "train_keys": list(train_keys),
        "hold_key": hold_key,
        "use_replay_last": bool(use_replay),
        "train_dirs": [str(p) for p in train_dirs],
        "hold_dirs": [str(p) for p in hold_dirs],
        "n_train_sessions": len(train_dirs),
    }


def _build_train_dirs(
    arm: str,
    base_train: Sequence[Path],
    *,
    seed: int,
    replay_ratio: Optional[float] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    base = [Path(p) for p in base_train]
    meta: Dict[str, Any] = {"arm": arm, "seed": seed}
    if arm == "A1":
        dirs = list(base)
        meta["rule"] = "full_history"
    elif arm == "A2":
        dirs = [base[-1]] if base else []
        meta["rule"] = "recent_only_1"
    elif arm == "A3":
        n = 1
        rng = random.Random(seed + 42)
        if len(base) <= n:
            dirs = list(base)
        else:
            dirs = rng.sample(list(base), n)
        meta["rule"] = "random_N"
        meta["N"] = n
        meta["picked"] = [p.name for p in dirs]
    elif arm == "A4":
        dirs = list(base)
        rng = random.Random(seed + 99)
        rng.shuffle(dirs)
        meta["rule"] = "shuffled_order"
        meta["order"] = [p.name for p in dirs]
    elif arm == "A5":
        dirs = list(base)
        meta["rule"] = "replay_sensitivity"
        meta["replay_ratio"] = float(replay_ratio if replay_ratio is not None else 0.10)
    else:
        raise ValueError(f"unknown arm {arm}")
    return dirs, meta


def _job_dir(person_id: str, arm: str, seed: int, replay_tag: str = "") -> Path:
    name = f"seed{seed}"
    if replay_tag:
        name = f"{replay_tag}_{name}"
    return A_ROOT / person_id / arm / name


def _metrics_path(out_dir: Path) -> Path:
    return out_dir / "exp42_metrics.json"


def _run_one(
    *,
    person_id: str,
    member_id: str,
    arm: str,
    seed: int,
    train_dirs: List[Path],
    hold_dirs: List[Path],
    replay_ratio: float,
    no_replay: bool,
    device: str,
    dry_run: bool,
    helpers: Dict[str, Any],
    arm_meta: Dict[str, Any],
) -> Dict[str, Any]:
    replay_tag = ""
    if arm == "A5":
        replay_tag = f"r{int(round(replay_ratio * 100)):02d}"
    out_dir = _job_dir(person_id, arm, seed, replay_tag=replay_tag)
    mpath = _metrics_path(out_dir)
    if mpath.is_file() and not dry_run:
        prev = json.loads(mpath.read_text(encoding="utf-8"))
        prev["skipped"] = True
        return prev

    payload = {
        "schema": "exp42_arm_A_job_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "person_id": person_id,
        "member_id": member_id,
        "arm": arm,
        "seed": seed,
        "out_dir": str(out_dir),
        "train_sessions": [p.name for p in train_dirs],
        "heldout_sessions": [p.name for p in hold_dirs],
        "replay_ratio": replay_ratio,
        "no_replay": no_replay,
        "arm_meta": arm_meta,
        "device": device,
    }
    if dry_run:
        payload["status"] = "dry_run"
        return payload

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job_spec.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import torch
        from experiment_game.pipeline.e1f_all4_ft import run_e1f_all4_finetune
        from adapt_engine.e1f import E1fStackConfig

        # Isolate from online current: empty models dir under out -> base stack init
        subject_models_dir = out_dir / "_init_models"
        subject_models_dir.mkdir(parents=True, exist_ok=True)

        result = run_e1f_all4_finetune(
            train_dirs,
            out_dir,
            subject_models_dir=subject_models_dir,
            heldout_session_dirs=hold_dirs,
            replay_ratio=float(replay_ratio),
            no_replay=bool(no_replay),
            seed=int(seed),
            early_stop=True,
            verbose=True,
            device=device,
        )
        gate = result.get("release_gate") or {}
        three = result.get("three") or {}

        # F5 trial-level MI on heldout with FT overlay (best-effort; never fail the job)
        mi_acc = float("nan")
        f5_err = None
        e1f_cfg = _REPO / "experiment_game" / "config" / "e1f_four_member.json"
        ov_path = out_dir / "e1f_overlay.json"
        try:
            if ov_path.is_file() and e1f_cfg.is_file():
                ov = json.loads(ov_path.read_text(encoding="utf-8"))
                ft_stack = (
                    helpers["E1fStackConfig"]
                    .load_json(e1f_cfg, repo_root=_REPO)
                    .with_member_overrides(ov.get("members") or {})
                    .resolve_paths(repo_root=_REPO)
                )
                ft_reg = helpers["E1fRegistry"](ft_stack, device=device)
                f5 = helpers["eval_f5_e1f"](hold_dirs, device=device, e1f_registry=ft_reg)
                mi_acc = _mi_from_f5_blob(f5)
        except Exception as exc:
            f5_err = str(exc)

        payload.update(
            {
                "status": "ok",
                "release_pass": bool(result.get("release_pass")),
                "heldout_acc_raw": _as_float(three.get("acc_after_heldout"), _as_float(gate.get("heldout_acc"))),
                "heldout_acc_smooth": _as_float(three.get("acc_after_heldout_smooth")),
                "max_class_frac": _as_float(gate.get("max_class_frac")),
                "mi_acc_f5": mi_acc,
                "f5_error": f5_err,
                "torch_cuda": bool(torch.cuda.is_available()),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc()[-2000:],
            }
        )

    mpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def iter_jobs(
    *,
    people: Optional[Sequence[str]],
    arms: Sequence[str],
    seeds: Sequence[int],
    helpers: Dict[str, Any],
) -> List[Dict[str, Any]]:
    cohort = _load_cohort()
    jobs: List[Dict[str, Any]] = []
    for person in cohort["people"]:
        pid = person["person_id"]
        mid = person["primary_id"]
        if people and pid not in people and mid not in people:
            continue
        stage = _last_stage(mid, helpers)
        if stage is None or stage.get("error"):
            jobs.append(
                {
                    "person_id": pid,
                    "member_id": mid,
                    "status": "skip",
                    "error": (stage or {}).get("error", "no_stage"),
                }
            )
            continue
        base_train = [Path(p) for p in stage["train_dirs"]]
        hold_dirs = [Path(p) for p in stage["hold_dirs"]]
        # Match last Leave-Next replay flag for A1-A4; A5 overrides
        base_replay = 0.1 if stage["use_replay_last"] else 0.0
        base_no_replay = not stage["use_replay_last"]
        for arm in arms:
            if arm == "A5":
                for rr in A5_REPLAY:
                    for seed in seeds:
                        train_dirs, meta = _build_train_dirs(
                            arm, base_train, seed=seed, replay_ratio=rr
                        )
                        jobs.append(
                            {
                                "person_id": pid,
                                "member_id": mid,
                                "arm": arm,
                                "seed": seed,
                                "train_dirs": train_dirs,
                                "hold_dirs": hold_dirs,
                                "replay_ratio": rr,
                                "no_replay": rr <= 0,
                                "arm_meta": meta,
                                "stage": stage,
                            }
                        )
            else:
                for seed in seeds:
                    train_dirs, meta = _build_train_dirs(arm, base_train, seed=seed)
                    jobs.append(
                        {
                            "person_id": pid,
                            "member_id": mid,
                            "arm": arm,
                            "seed": seed,
                            "train_dirs": train_dirs,
                            "hold_dirs": hold_dirs,
                            "replay_ratio": base_replay,
                            "no_replay": base_no_replay,
                            "arm_meta": meta,
                            "stage": stage,
                        }
                    )
    return jobs


def aggregate_A(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute R_vol / R_span / R_order from completed A1-A4 jobs."""
    # key: (person, seed) -> arm -> mi
    by: Dict[Tuple[str, int], Dict[str, float]] = {}
    for r in results:
        if r.get("status") not in ("ok",) and not r.get("skipped"):
            continue
        if r.get("arm") not in ("A1", "A2", "A3", "A4"):
            continue
        mi = r.get("mi_acc_f5")
        if mi is None or not np.isfinite(float(mi)):
            # fallback to smooth window
            mi = r.get("heldout_acc_smooth")
        if mi is None or not np.isfinite(float(mi)):
            continue
        key = (r["person_id"], int(r["seed"]))
        by.setdefault(key, {})[r["arm"]] = float(mi)

    per_person: Dict[str, Dict[str, List[float]]] = {}
    for (pid, _seed), arms in by.items():
        bucket = per_person.setdefault(
            pid, {"R_vol": [], "R_span": [], "R_order": [], "A1": [], "A2": [], "A3": [], "A4": []}
        )
        if "A1" in arms and "A2" in arms:
            bucket["R_vol"].append(arms["A1"] - arms["A2"])
            bucket["A1"].append(arms["A1"])
            bucket["A2"].append(arms["A2"])
        if "A3" in arms and "A2" in arms:
            bucket["R_span"].append(arms["A3"] - arms["A2"])
            bucket["A3"].append(arms["A3"])
        if "A1" in arms and "A4" in arms:
            bucket["R_order"].append(arms["A1"] - arms["A4"])
            bucket["A4"].append(arms["A4"])

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    people_rows = []
    rvol_means = []
    for pid, b in sorted(per_person.items()):
        row = {
            "person_id": pid,
            "R_vol_mean": _mean(b["R_vol"]),
            "R_span_mean": _mean(b["R_span"]),
            "R_order_mean": _mean(b["R_order"]),
            "A1_mean": _mean(b["A1"]),
            "A2_mean": _mean(b["A2"]),
            "n_seed_R_vol": len(b["R_vol"]),
        }
        people_rows.append(row)
        if np.isfinite(row["R_vol_mean"]):
            rvol_means.append(row["R_vol_mean"])

    n = len(rvol_means) or 1
    frac_pos = sum(1 for x in rvol_means if x > 0) / n if rvol_means else float("nan")
    mean_vol = _mean(rvol_means)
    if mean_vol >= 0.03 and frac_pos >= 2 / 3:
        verdict = "volume_dominant"
    elif mean_vol < 0.01:
        verdict = "recent_state_dominant"
    else:
        verdict = "mixed_or_inconclusive"

    # A5 damping: slope of mi vs replay ratio per person
    a5_by: Dict[str, List[Tuple[float, float]]] = {}
    for r in results:
        if r.get("arm") != "A5":
            continue
        if r.get("status") not in ("ok",) and not r.get("skipped"):
            continue
        mi = r.get("mi_acc_f5")
        if mi is None or not np.isfinite(float(mi)):
            mi = r.get("heldout_acc_smooth")
        if mi is None or not np.isfinite(float(mi)):
            continue
        rr = float(r.get("replay_ratio") or float("nan"))
        if not np.isfinite(rr):
            continue
        a5_by.setdefault(r["person_id"], []).append((rr, float(mi)))
    dA5 = []
    for pid, pts in a5_by.items():
        # average across seeds per ratio
        from collections import defaultdict

        acc: Dict[float, List[float]] = defaultdict(list)
        for rr, mi in pts:
            acc[rr].append(mi)
        xs = sorted(acc.keys())
        if len(xs) < 2:
            continue
        ys = [float(np.mean(acc[x])) for x in xs]
        slope = float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])
        dA5.append({"person_id": pid, "dA5_slope": slope, "points": list(zip(xs, ys))})

    return {
        "schema": "exp42_arm_A_aggregate_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_people": len(people_rows),
        "R_vol_mean": mean_vol,
        "frac_positive_R_vol": frac_pos,
        "verdict": verdict,
        "dA5": dA5,
        "dA5_mean_slope": float(np.mean([d["dA5_slope"] for d in dA5])) if dA5 else float("nan"),
        "per_person": people_rows,
    }


def run_arm_A(
    *,
    people: Optional[Sequence[str]] = None,
    arms: Sequence[str] = DEFAULT_ARMS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: Optional[str] = None,
    dry_run: bool = False,
    max_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    helpers = _import_leave_next()
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    jobs = iter_jobs(people=people, arms=arms, seeds=seeds, helpers=helpers)
    runnable = [j for j in jobs if "arm" in j]
    if max_jobs is not None:
        runnable = runnable[: max(0, int(max_jobs))]

    A_ROOT.mkdir(parents=True, exist_ok=True)
    queue_path = A_ROOT / "job_queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "n_jobs": len(runnable),
                "device": device,
                "dry_run": dry_run,
                "jobs": [
                    {
                        "person_id": j["person_id"],
                        "member_id": j["member_id"],
                        "arm": j["arm"],
                        "seed": j["seed"],
                        "replay_ratio": j["replay_ratio"],
                        "train": [p.name for p in j["train_dirs"]],
                        "hold": [p.name for p in j["hold_dirs"]],
                    }
                    for j in runnable
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[A] queue={len(runnable)} device={device} dry_run={dry_run} -> {queue_path}")

    results: List[Dict[str, Any]] = []
    for i, j in enumerate(runnable, 1):
        print(
            f"\n[A {i}/{len(runnable)}] {j['person_id']} {j['arm']} seed={j['seed']} "
            f"replay={j['replay_ratio']}"
        )
        results.append(
            _run_one(
                person_id=j["person_id"],
                member_id=j["member_id"],
                arm=j["arm"],
                seed=int(j["seed"]),
                train_dirs=j["train_dirs"],
                hold_dirs=j["hold_dirs"],
                replay_ratio=float(j["replay_ratio"]),
                no_replay=bool(j["no_replay"]),
                device=device,
                dry_run=dry_run,
                helpers=helpers,
                arm_meta=j["arm_meta"],
            )
        )

    # also collect already-finished metrics under A_ROOT for aggregate
    if not dry_run:
        for p in A_ROOT.rglob("exp42_metrics.json"):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if obj.get("status") in ("ok",) or obj.get("skipped"):
                # avoid dupes
                if not any(
                    r.get("out_dir") == obj.get("out_dir") and r.get("arm") == obj.get("arm")
                    for r in results
                ):
                    results.append(obj)

    agg = aggregate_A(results) if not dry_run else {"status": "dry_run", "n_jobs": len(runnable)}
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out_agg = ANALYSIS / "arm_A_aggregate.json"
    out_agg.write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (A_ROOT / "aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "exp42_arm_A_run_summary_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_submitted": len(runnable),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "n_skip_cached": sum(1 for r in results if r.get("skipped")),
        "aggregate_path": str(out_agg),
        "verdict": agg.get("verdict"),
        "R_vol_mean": agg.get("R_vol_mean"),
    }
    (A_ROOT / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[A] done ok={summary['n_ok']} err={summary['n_error']} verdict={summary.get('verdict')}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp42 Arm A jackknife")
    ap.add_argument("--people", type=str, default="", help="comma person_ids; empty=all")
    ap.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=0, help="0=all; >0 smoke limit")
    args = ap.parse_args()
    people = [x.strip() for x in args.people.split(",") if x.strip()] or None
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    run_arm_A(
        people=people,
        arms=arms,
        seeds=seeds,
        device=args.device or None,
        dry_run=args.dry_run,
        max_jobs=args.max_jobs or None,
    )


if __name__ == "__main__":
    main()
