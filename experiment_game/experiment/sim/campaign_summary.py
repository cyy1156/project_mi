"""Campaign 结束 / 增量汇总报告。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_game.experiment.sim.campaign import campaign_summary_path, load_campaign, save_campaign


def append_session_result(
    manifest: Dict[str, Any],
    *,
    session_dir: Path,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """向 manifest 追加一场 session 结果。"""
    rec: Dict[str, Any] = {
        "session_dir": str(session_dir.resolve()),
        "run_id": None,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = session_dir / "session.meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rec["run_id"] = meta.get("source_run") or meta.get("session_id")
            rec["session_trials_total"] = meta.get("session_trials_total")
            rec["trials_unused"] = meta.get("trials_unused")
        except Exception:
            pass
    if summary:
        rec["session_score"] = summary.get("session_score")
        rec["session_score_max"] = summary.get("session_score_max")
        rep = summary.get("report") or {}
        overall = rep.get("overall") or {}
        rec["acc_argmax"] = overall.get("acc_argmax")
        rec["quality_tier"] = summary.get("quality_tier")
    report_path = session_dir / "v3_report.json"
    if report_path.is_file():
        try:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            ov = rep.get("overall") or {}
            rec["acc_argmax"] = rec.get("acc_argmax") or ov.get("acc_argmax")
        except Exception:
            pass

    done = list(manifest.get("sessions_completed") or [])
    done.append(rec)
    manifest["sessions_completed"] = done
    save_campaign(manifest)
    return rec


def exclude_session_from_records(
    session_dir: Path,
    *,
    campaign_manifest: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
    reason: str = "operator_exclude",
) -> Dict[str, Any]:
    """标记 session 不写入实验记录；文件保留，仍可用于 FT。

    - session.meta.json: record_excluded=True
    - Campaign: 从 sessions_completed 移除，run 从 runs_consumed 释放以便重采
    """
    session_dir = Path(session_dir).resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"session 目录不存在: {session_dir}")

    meta_path = session_dir / "session.meta.json"
    run_id: Optional[str] = None
    subject_id: Optional[str] = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("record_excluded"):
            return {
                "ok": True,
                "already_excluded": True,
                "session_dir": str(session_dir),
                "run_id": meta.get("source_run") or meta.get("session_id"),
            }
        run_id = str(meta.get("source_run") or meta.get("session_id") or "").strip().lower() or None
        subject_id = meta.get("subject_id")
        meta["record_excluded"] = True
        meta["record_excluded_at"] = datetime.now().isoformat(timespec="seconds")
        meta["record_excluded_reason"] = reason
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    campaign_updated = False
    summary_path: Optional[str] = None
    next_run: Optional[str] = None
    if campaign_manifest is not None:
        manifest = dict(campaign_manifest)
        mp = manifest.get("manifest_path")
        if mp:
            manifest = load_campaign(mp)
        done = list(manifest.get("sessions_completed") or [])
        new_done = [
            r
            for r in done
            if Path(str(r.get("session_dir") or "")).resolve() != session_dir
        ]
        removed = len(done) - len(new_done)
        manifest["sessions_completed"] = new_done
        if run_id:
            consumed = set(manifest.get("runs_consumed") or [])
            if run_id in consumed:
                consumed.discard(run_id)
                manifest["runs_consumed"] = sorted(consumed)
                queue = list(manifest.get("session_queue") or [])
                if run_id in queue:
                    manifest["current_index"] = queue.index(run_id)
                manifest["status"] = "active"
        save_campaign(manifest)
        summary_path = str(write_campaign_summary(manifest, repo_root=repo_root))
        remaining = [
            r
            for r in (manifest.get("session_queue") or [])
            if r not in set(manifest.get("runs_consumed") or [])
        ]
        next_run = remaining[0] if remaining else None
        campaign_updated = removed > 0 or bool(run_id)
        campaign_manifest = manifest

    sim_index = None
    if subject_id:
        try:
            from experiment_game.experiment.sim.sim_registry import (
                build_sim_index,
                validate_sim_subject_id,
            )

            sid = validate_sim_subject_id(str(subject_id))
            sim_index = build_sim_index(sid, repo_root=repo_root)
        except Exception:
            pass

    return {
        "ok": True,
        "session_dir": str(session_dir),
        "run_id": run_id,
        "campaign_updated": campaign_updated,
        "campaign": campaign_manifest,
        "campaign_summary_path": summary_path,
        "next_run": next_run,
        "sim_index": sim_index,
        "subject_id": subject_id,
    }


def write_campaign_summary(
    manifest: Dict[str, Any],
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """生成 records/campaigns/<id>/summary.md 与 analysis 副本。"""
    from experiment_game.experiment.sim.sim_registry import sim_subject_root

    sid = manifest.get("subject_id") or "A01"
    lines = [
        f"# Campaign 汇总 · {sid} · {manifest.get('campaign_id')}",
        "",
        f"- 创建：{manifest.get('created_at')}",
        f"- 状态：{manifest.get('status')}",
        f"- 队列：{' → '.join(manifest.get('session_queue') or [])}",
        f"- 已用 run：{', '.join(manifest.get('runs_consumed') or [])}",
        f"- 每场 trial：{manifest.get('session_trials_total')}",
        f"- 回放：{manifest.get('replay_align')} @ {manifest.get('replay_speed')}×",
        "",
        "## 各场 session",
        "",
        "| run | 得分 | acc | quality | 目录 |",
        "|-----|------|-----|---------|------|",
    ]
    for rec in manifest.get("sessions_completed") or []:
        run = rec.get("run_id") or "—"
        score = rec.get("session_score")
        smax = rec.get("session_score_max")
        sc = f"{score}/{smax}" if score is not None and smax else "—"
        acc = rec.get("acc_argmax")
        acc_s = f"{float(acc):.3f}" if acc is not None else "—"
        qt = rec.get("quality_tier") or "—"
        d = Path(str(rec.get("session_dir") or "")).name
        lines.append(f"| {run} | {sc} | {acc_s} | {qt} | `{d}` |")

    remaining = [
        r
        for r in (manifest.get("session_queue") or [])
        if r not in set(manifest.get("runs_consumed") or [])
    ]
    if remaining:
        lines.extend(["", f"**剩余 run：** {', '.join(remaining)}"])
    else:
        lines.extend(["", "**Campaign 队列已全部完成。**"])

    text = "\n".join(lines) + "\n"
    out = campaign_summary_path(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    analysis = sim_subject_root(sid, repo_root=repo_root) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    copy = analysis / f"campaign_{manifest.get('campaign_id')}_summary.md"
    copy.write_text(text, encoding="utf-8")
    return out
