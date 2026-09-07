"""Exp29 Leave-Next-Run Ramp 辅助（Campaign · FT 选 session）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

T0_THREE_REL = (
    "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper"
    "/shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100"
    "/run_20260822_094942/three/fold0/best_three.pt"
)
T0_TASK_REL = T0_THREE_REL.replace("/three/", "/task/")


def parse_manifest(manifest: Dict[str, Any] | Path | str) -> Dict[str, Any]:
    if isinstance(manifest, (Path, str)):
        return json.loads(Path(manifest).read_text(encoding="utf-8"))
    return manifest


def queue_index(manifest: Dict[str, Any], run_id: str) -> int:
    q = [str(r).strip().lower() for r in manifest.get("session_queue") or []]
    rid = str(run_id).strip().lower()
    if rid not in q:
        raise ValueError(f"run {run_id} 不在 Campaign 队列")
    return q.index(rid)


def completed_by_run(manifest: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in manifest.get("sessions_completed") or []:
        rid = str(item.get("run_id") or "").strip().lower()
        path = str(item.get("session_dir") or "").strip()
        if rid and path:
            out[rid] = path
    return out


def leave_next_train_runs(
    manifest: Dict[str, Any],
    eval_run_id: str,
) -> List[Tuple[str, str]]:
    """评估 run 之前的队列 run → 应用作 FT 训练 session（Leave-Next）。"""
    m = parse_manifest(manifest)
    q = [str(r).strip().lower() for r in m.get("session_queue") or []]
    eval_run_id = str(eval_run_id).strip().lower()
    if eval_run_id not in q:
        return []
    idx = q.index(eval_run_id)
    train_ids = q[:idx]
    done = completed_by_run(m)
    return [(rid, done[rid]) for rid in train_ids if rid in done]


def ramp_stage(manifest: Dict[str, Any], eval_run_id: str) -> int:
    """R0=仅底座评估 run3；R1=1 run FT …"""
    m = parse_manifest(manifest)
    q = [str(r).strip().lower() for r in m.get("session_queue") or []]
    eval_run_id = str(eval_run_id).strip().lower()
    if eval_run_id not in q:
        return 0
    return q.index(eval_run_id)


def next_eval_run(manifest: Dict[str, Any]) -> Optional[str]:
    m = parse_manifest(manifest)
    q = [str(r).strip().lower() for r in m.get("session_queue") or []]
    done = set(completed_by_run(m).keys())
    for rid in q:
        if rid not in done:
            return rid
    return None


def ft_replay_recommendation(r_stage: int) -> Dict[str, Any]:
    """Exp29 B2：R1–R3 开 replay；R≥4 建议关。"""
    if r_stage <= 0:
        return {"use_replay": False, "replay_ratio": 0.0, "reason": "R0 不 FT"}
    if r_stage >= 4:
        return {
            "use_replay": False,
            "replay_ratio": 0.0,
            "reason": f"R{r_stage} 窗数充足，建议纯被试窗 FT（Exp29）",
        }
    return {
        "use_replay": True,
        "replay_ratio": 0.10,
        "reason": f"R{r_stage} 数据偏少，建议 T0 replay 0.10（B2）",
    }


def weight_preset_for_ramp(r_stage: int, *, repo_root: Optional[Path] = None) -> Dict[str, str]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    if r_stage <= 0:
        return {
            "s3_three_ckpt": str((root / T0_THREE_REL).resolve()),
            "s3_task_ckpt": str((root / T0_TASK_REL).resolve()),
            "label": "T0 底座",
        }
    return {"label": "上一轮 promote 的 ft_runs / current"}
