"""同协议配对 run 查找（v1.2 严格口径）。"""

from __future__ import annotations

import json
from pathlib import Path

from config import ADABN_VERSION, PROTOCOL_VERSION, RESULTS_ROOT


def load_paired_summary(
    anchor_arm: str,
    *,
    head: str = "three",
    results_root: Path | None = None,
) -> dict | None:
    """读取锚点臂最新合格 summary；返回 per_subject 列表与路径。"""
    base_dir = (results_root or RESULTS_ROOT) / f"S09-{anchor_arm}"
    if not base_dir.is_dir():
        return None
    for path in sorted(base_dir.glob("*/summary.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        spec = data.get("spec") or {}
        if spec.get("protocol_version") != PROTOCOL_VERSION:
            continue
        if spec.get("adabn_version") != ADABN_VERSION:
            continue
        if spec.get("eval_protocol") != "eval_half_causal":
            continue
        if spec.get("input_pipeline") != "noz_unified":
            continue
        macro = data.get("macro") or {}
        block = macro.get(head) or {}
        per = block.get("per_subject")
        if per:
            return {
                "path": str(path),
                "per_subject": per,
                "mean": block.get("mean"),
                "by_subject": _per_subject_map(data, head),
            }
    return None


def _per_subject_map(summary: dict, head: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for sid, sub in (summary.get("subjects") or {}).items():
        m = (sub.get("metrics") or {}).get(head) or {}
        v = m.get("acc_paper_mean")
        if v is not None:
            out[str(sid)] = float(v)
    return out


def subject_acc(paired: dict | None, subject_id: str) -> float | None:
    if not paired:
        return None
    by = paired.get("by_subject") or {}
    return by.get(subject_id)
