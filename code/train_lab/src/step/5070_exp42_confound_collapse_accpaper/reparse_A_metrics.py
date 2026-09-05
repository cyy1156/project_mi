# -*- coding: utf-8 -*-
"""Rebuild exp42_metrics.json from finished A-job artifacts (no retrain).

Reads job_spec.json + release_gate.json + meta.json under out/.../A/.
Optionally re-runs F5 MI eval (--eval-f5) with the correct nested mi_acc path.

Usage:
  python reparse_A_metrics.py
  python reparse_A_metrics.py --eval-f5 --device cuda
  python reparse_A_metrics.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from paths import ANALYSIS, OUT_ROOT, REPO  # noqa: E402
from arm_A_jackknife import (  # noqa: E402
    A_ROOT,
    _as_float,
    _import_leave_next,
    _mi_from_f5_blob,
    aggregate_A,
)

SUBJECTS = REPO / "experiment_game" / "data" / "subjects"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _reparse_one(
    job_dir: Path,
    *,
    eval_f5: bool,
    device: str,
    helpers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    spec = _load_json(job_dir / "job_spec.json") or {}
    gate = _load_json(job_dir / "release_gate.json") or {}
    meta = _load_json(job_dir / "meta.json") or {}
    prev = _load_json(job_dir / "exp42_metrics.json") or {}

    person_id = spec.get("person_id") or prev.get("person_id") or job_dir.parent.parent.name
    arm = spec.get("arm") or prev.get("arm") or job_dir.parent.name
    seed = int(spec.get("seed") if spec.get("seed") is not None else prev.get("seed") or 0)
    replay_ratio = _as_float(spec.get("replay_ratio"), _as_float(prev.get("replay_ratio"), 0.0))

    raw = _as_float(meta.get("fusion_heldout_acc"), _as_float(gate.get("heldout_acc")))
    smooth = _as_float(meta.get("fusion_heldout_acc_smooth"), _as_float(gate.get("heldout_acc_smooth")))
    mcf = _as_float(gate.get("max_class_frac"))
    release_pass = bool(gate.get("pass")) if "pass" in gate else bool(meta.get("status") == "PASS")

    mi_acc = _as_float(prev.get("mi_acc_f5"))
    f5_err = None
    if eval_f5 and helpers is not None:
        hold_names = spec.get("heldout_sessions") or prev.get("heldout_sessions") or meta.get("heldout_sessions") or []
        member_id = spec.get("member_id") or prev.get("member_id") or person_id
        hold_dirs: List[Path] = []
        sess_root = SUBJECTS / str(member_id) / "sessions"
        for name in hold_names:
            p = sess_root / name
            if p.is_dir():
                hold_dirs.append(p)
        ov_path = job_dir / "e1f_overlay.json"
        e1f_cfg = REPO / "experiment_game" / "config" / "e1f_four_member.json"
        try:
            if hold_dirs and ov_path.is_file() and e1f_cfg.is_file():
                ov = json.loads(ov_path.read_text(encoding="utf-8"))
                ft_stack = (
                    helpers["E1fStackConfig"]
                    .load_json(e1f_cfg, repo_root=REPO)
                    .with_member_overrides(ov.get("members") or {})
                    .resolve_paths(repo_root=REPO)
                )
                ft_reg = helpers["E1fRegistry"](ft_stack, device=device)
                f5 = helpers["eval_f5_e1f"](hold_dirs, device=device, e1f_registry=ft_reg)
                mi_acc = _mi_from_f5_blob(f5)
        except Exception as exc:
            f5_err = str(exc)
            traceback.print_exc()

    # Prefer existing FT artifacts: status ok if gate/meta present
    ok = bool(gate) or bool(meta.get("fusion_heldout_acc") is not None)
    payload = {
        "schema": "exp42_arm_A_job_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "person_id": person_id,
        "member_id": spec.get("member_id") or prev.get("member_id") or person_id,
        "arm": arm,
        "seed": seed,
        "out_dir": str(job_dir),
        "train_sessions": spec.get("train_sessions") or prev.get("train_sessions") or meta.get("sessions"),
        "heldout_sessions": spec.get("heldout_sessions") or prev.get("heldout_sessions") or meta.get("heldout_sessions"),
        "replay_ratio": replay_ratio,
        "no_replay": bool(spec.get("no_replay") if "no_replay" in spec else prev.get("no_replay")),
        "arm_meta": spec.get("arm_meta") or prev.get("arm_meta"),
        "device": spec.get("device") or prev.get("device"),
        "status": "ok" if ok else "error",
        "release_pass": release_pass,
        "heldout_acc_raw": raw,
        "heldout_acc_smooth": smooth,
        "max_class_frac": mcf,
        "mi_acc_f5": mi_acc,
        "f5_error": f5_err,
        "reparsed_from_disk": True,
        "metric_primary_note": "R_* uses mi_acc_f5 if finite else heldout_acc_smooth (window)",
    }
    if not ok:
        payload["error"] = "missing release_gate/meta"
    (job_dir / "exp42_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def reparse_all(*, eval_f5: bool, device: str, limit: int = 0) -> List[Dict[str, Any]]:
    helpers = None
    if eval_f5:
        helpers = _import_leave_next()
    jobs = sorted(A_ROOT.rglob("job_spec.json"))
    results: List[Dict[str, Any]] = []
    for i, spec_path in enumerate(jobs, 1):
        if limit and i > limit:
            break
        job_dir = spec_path.parent
        print(f"[reparse {i}/{len(jobs) if not limit else limit}] {job_dir.relative_to(A_ROOT)}")
        results.append(_reparse_one(job_dir, eval_f5=eval_f5, device=device, helpers=helpers))
    return results


def write_aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = aggregate_A(results)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    A_ROOT.mkdir(parents=True, exist_ok=True)
    out_agg = ANALYSIS / "arm_A_aggregate.json"
    out_agg.write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (A_ROOT / "aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "exp42_arm_A_run_summary_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_submitted": len(results),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "n_skip_cached": 0,
        "aggregate_path": str(out_agg),
        "verdict": agg.get("verdict"),
        "R_vol_mean": agg.get("R_vol_mean"),
        "R_span_mean": float(
            __import__("numpy").nanmean([p.get("R_span_mean") for p in agg.get("per_person") or []])
        )
        if agg.get("per_person")
        else None,
        "reparsed": True,
    }
    (A_ROOT / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[aggregate] ok={summary['n_ok']} err={summary['n_error']} "
        f"verdict={summary.get('verdict')} R_vol={summary.get('R_vol_mean')}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-f5", action="store_true", help="recompute trial-level mi_acc_f5")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.aggregate_only:
        results = []
        for p in A_ROOT.rglob("exp42_metrics.json"):
            d = _load_json(p)
            if d:
                results.append(d)
        write_aggregate(results)
    else:
        results = reparse_all(eval_f5=args.eval_f5, device=device, limit=args.limit)
        write_aggregate(results)

    # refresh registry
    from run_all import stage_summary

    stage_summary()
    print("[reparse] DONE")


if __name__ == "__main__":
    main()
