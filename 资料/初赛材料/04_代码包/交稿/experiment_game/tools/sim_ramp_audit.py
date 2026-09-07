"""Campaign Leave-Next Ramp 审计：L/R acc · 预测分布 · FT 窗数 · 权重。

用法:
  python experiment_game/tools/sim_ramp_audit.py \\
    --manifest experiment_game/data/sim_subjects/A01/records/campaigns/sim_xxx/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from experiment_game.experiment.sim.ramp import (  # noqa: E402
    completed_by_run,
    ft_replay_recommendation,
    leave_next_train_runs,
    ramp_stage,
)
from experiment_game.tools.ft_subject_from_v3 import (  # noqa: E402
    build_dataset,
    is_sim_session,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _trial_lr_stats(session_dir: Path) -> Dict[str, Any]:
    recs = _load_jsonl(Path(session_dir) / "v3_trial_features.jsonl")
    lr = [
        r
        for r in recs
        if int(r.get("label") or -1) in (1, 2)
        and r.get("valid")
        and not r.get("signal_bad")
    ]
    preds: List[int] = []
    correct = 0
    for r in lr:
        pj = r.get("primary_judge") or {}
        pred = pj.get("pred")
        if pred is None:
            gated = pj.get("gated_pred")
            if gated is not None:
                pred = gated
        if pred is None:
            continue
        pred_i = int(pred)
        preds.append(pred_i)
        if pred_i == int(r["label"]):
            correct += 1
    acc = correct / len(lr) if lr else None
    return {
        "n_lr_valid": len(lr),
        "lr_acc": acc,
        "pred_dist": dict(Counter(preds)),
    }


def _session_weights(session_dir: Path) -> Dict[str, Any]:
    meta_p = Path(session_dir) / "session.meta.json"
    if not meta_p.is_file():
        return {}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    eff = meta.get("v3_config_effective") or {}
    return {
        "s3_three_ckpt": eff.get("s3_three_ckpt") or meta.get("s3_three_ckpt"),
        "s3_task_ckpt": eff.get("s3_task_ckpt") or meta.get("s3_task_ckpt"),
    }


def _ft_window_count(session_dir: Path, *, exclude_invalid: bool = False) -> Optional[int]:
    sd = Path(session_dir)
    if not is_sim_session(sd):
        return None
    try:
        ds = build_dataset(sd, include_invalid=not exclude_invalid, protocol="auto")
        return int(ds["X"].shape[0])
    except Exception as exc:  # noqa: BLE001
        return -1


def _alignment_cue_coverage(session_dir: Path) -> Dict[str, Any]:
    table_p = Path(session_dir) / "alignment" / "trial_table.csv"
    if not table_p.is_file():
        return {"n_rows": 0, "n_with_cue": 0}
    df = pd.read_csv(table_p)
    n_with = int(df["t_cue"].notna().sum()) if "t_cue" in df.columns else 0
    return {"n_rows": len(df), "n_with_cue": n_with}


def audit_campaign(manifest_path: Path, *, exclude_invalid: bool = False) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queue = [str(r).strip().lower() for r in manifest.get("session_queue") or []]
    done = completed_by_run(manifest)
    rows: List[Dict[str, Any]] = []

    for run_id in queue:
        session_dir = done.get(run_id)
        row: Dict[str, Any] = {
            "run_id": run_id,
            "r_stage": ramp_stage(manifest, run_id),
            "session_dir": session_dir,
            "completed": session_dir is not None,
        }
        if session_dir:
            row.update(_trial_lr_stats(session_dir))
            row["weights"] = _session_weights(session_dir)
            row["ft_windows"] = _ft_window_count(session_dir, exclude_invalid=exclude_invalid)
            row["alignment"] = _alignment_cue_coverage(session_dir)
            train = leave_next_train_runs(manifest, run_id)
            row["leave_next_train"] = [rid for rid, _ in train]
            row["leave_next_train_dirs"] = [p for _, p in train]
            row["ft_replay_rec"] = ft_replay_recommendation(row["r_stage"])
        rows.append(row)

    return {
        "campaign_id": manifest.get("campaign_id"),
        "subject_id": manifest.get("subject_id"),
        "leave_next_mode": manifest.get("leave_next_mode", True),
        "replay_speed": manifest.get("replay_speed"),
        "runs": rows,
    }


def _format_report(audit: Dict[str, Any]) -> str:
    lines = [
        f"# Ramp Audit · {audit.get('campaign_id')} · {audit.get('subject_id')}",
        "",
        f"- Leave-Next: {audit.get('leave_next_mode')}",
        f"- replay_speed: {audit.get('replay_speed')}",
        "",
        "| run | R | L/R acc | pred | FT窗 | train runs | replay |",
        "|-----|---|---------|------|------|------------|--------|",
    ]
    for r in audit.get("runs") or []:
        if not r.get("completed"):
            lines.append(f"| {r['run_id']} | R{r['r_stage']} | — | — | — | — | — |")
            continue
        acc = r.get("lr_acc")
        acc_s = f"{acc * 100:.0f}%" if acc is not None else "—"
        pred = r.get("pred_dist") or {}
        pred_s = ",".join(f"{k}:{v}" for k, v in sorted(pred.items())) or "—"
        fw = r.get("ft_windows")
        fw_s = str(fw) if fw is not None and fw >= 0 else "ERR"
        train = ",".join(r.get("leave_next_train") or []) or "—"
        rep = r.get("ft_replay_rec") or {}
        rep_s = "on" if rep.get("use_replay") else "off"
        lines.append(
            f"| {r['run_id']} | R{r['r_stage']} | {acc_s} | {pred_s} | {fw_s} | {train} | {rep_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Campaign Leave-Next Ramp 审计")
    ap.add_argument("--manifest", type=Path, required=True, help="Campaign manifest.json")
    ap.add_argument("--exclude-invalid", action="store_true")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    audit = audit_campaign(args.manifest.resolve(), exclude_invalid=args.exclude_invalid)
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print(_format_report(audit))


if __name__ == "__main__":
    main()
