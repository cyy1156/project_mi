"""同协议配对 run 查找（v1.2 严格口径）。"""

from __future__ import annotations

import json
from pathlib import Path

from config import ADABN_VERSION, PROTOCOL_VERSION, RESULTS_ROOT

EXPECTED_N_SUBJECTS = 24


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


def paired_subject_count(paired: dict | None) -> int:
    if not paired:
        return 0
    by = paired.get("by_subject") or {}
    return len(by)


def require_paired_summaries(
    anchor_arms: tuple[str, ...] = ("A0", "B3"),
    *,
    min_subjects: int = EXPECTED_N_SUBJECTS,
    head: str = "three",
    results_root: Path | None = None,
) -> dict[str, dict]:
    """C1 前校验：锚点臂须为 v1.2 全量 run，且覆盖足够被试。"""
    missing: list[str] = []
    partial: list[str] = []
    out: dict[str, dict] = {}
    for arm in anchor_arms:
        paired = load_paired_summary(arm, head=head, results_root=results_root)
        if not paired:
            missing.append(arm)
            continue
        n = paired_subject_count(paired)
        if n < min_subjects:
            partial.append(f"{arm}({n}/{min_subjects})")
            continue
        out[arm] = paired
    if missing or partial:
        lines = ["C1 需要同链 v1.2 全量锚点 summary，请先跑 eval_ab："]
        if missing:
            lines.append(f"  缺失: {', '.join(missing)}")
        if partial:
            lines.append(f"  被试不足: {', '.join(partial)}")
        lines.append("  冒烟仅: eval_c1.py --smoke --subjects S1")
        raise RuntimeError("\n".join(lines))
    return out
