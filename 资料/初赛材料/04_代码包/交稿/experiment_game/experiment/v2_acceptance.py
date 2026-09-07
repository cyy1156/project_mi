"""v2 收工验收与 phase4 管线窗数统计（操作台 Summary）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def count_phase4_windows(session_root: Path) -> Dict[str, Any]:
    """统计 phase4_v2 / phase4_v2_game / pooled 窗数。"""
    root = Path(session_root)
    out: Dict[str, Any] = {"pipes": {}}
    for name in ("phase4_v2", "phase4_v2_game", "phase4_v2_pooled"):
        d = root / name
        meta = {}
        n = None
        x_path = d / "X.npy"
        man = d / "manifest.json"
        if man.is_file():
            try:
                meta = json.loads(man.read_text(encoding="utf-8"))
                n = meta.get("n_windows")
            except Exception:
                pass
        if n is None and x_path.is_file():
            try:
                import numpy as np

                n = int(np.load(x_path, mmap_mode="r").shape[0])
            except Exception:
                n = None
        out["pipes"][name] = {
            "exists": d.is_dir(),
            "n_windows": n,
            "path": str(d) if d.is_dir() else None,
            "manifest": str(man) if man.is_file() else None,
        }
    merged = root / "phase4_v2_merged" / "manifest.json"
    if not merged.is_file():
        merged = root / "phase4_merged" / "manifest.json"
    out["merged_manifest"] = str(merged) if merged.is_file() else None
    return out


def compute_v2_acceptance(
    *,
    summary: Optional[Dict[str, Any]],
    verify: Optional[Dict[str, Any]],
    session_root: Optional[Path] = None,
    min_valid_trials: int = 88,
    balance_tol: float = 0.15,
) -> Dict[str, Any]:
    """采集流程 §7 四门：有效试次 / 类别均衡 / 准入记录 / alignment。

    返回 level: green|yellow|red|na 与各门明细。
    """
    summary = summary or {}
    verify = verify or {}
    gates: List[Dict[str, Any]] = []

    n_valid = int(summary.get("valid_trials") or 0)
    g_valid = {
        "id": "valid_trials",
        "ok": n_valid >= min_valid_trials,
        "detail": f"有效试次 {n_valid}（门限 ≥{min_valid_trials}）",
    }
    gates.append(g_valid)

    labels = summary.get("labels") or {}
    counts = {0: 0, 1: 0, 2: 0}
    for v in labels.values():
        try:
            counts[int(v)] = counts.get(int(v), 0) + 1
        except (TypeError, ValueError):
            pass
    total = sum(counts.values()) or 1
    nominal = 1.0 / 3.0
    bal_ok = all(abs(counts[c] / total - nominal) <= balance_tol for c in (0, 1, 2)) if total >= 3 else False
    gates.append({
        "id": "class_balance",
        "ok": bal_ok,
        "detail": f"类别计数 L/R/Rest={counts.get(1,0)}/{counts.get(2,0)}/{counts.get(0,0)}（标称±{int(balance_tol*100)}%）",
    })

    gate_status = summary.get("gate_status")
    gate_ok = gate_status in ("pass", "weak_mi", "extend", "fail_pending", "skipped", "degraded")
    curve = summary.get("curve") or []
    gates.append({
        "id": "gate_record",
        "ok": bool(gate_ok and (curve or gate_status in ("skipped", "degraded", "pass", "weak_mi"))),
        "detail": f"准入 status={gate_status} · 曲线点 {len(curve)}",
    })

    align_ok = verify.get("passed") is True
    gates.append({
        "id": "alignment",
        "ok": align_ok,
        "detail": "alignment passed" if align_ok else f"alignment={verify.get('passed')}",
    })

    n_ok = sum(1 for g in gates if g["ok"])
    if n_ok == 4:
        level = "green"
    elif n_ok >= 2:
        level = "yellow"
    else:
        level = "red"
    if summary.get("degraded") and n_valid == 0:
        level = "na"

    return {
        "level": level,
        "gates": gates,
        "n_ok": n_ok,
        "n_total": 4,
        "phase4": count_phase4_windows(session_root) if session_root else {},
    }
