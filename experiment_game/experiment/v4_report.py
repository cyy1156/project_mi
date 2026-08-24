"""v4 会话报告落盘。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_v4_report(
    session_dir: Path,
    summary: Dict[str, Any],
    *,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    session_dir = Path(session_dir)
    payload = {
        "summary": summary,
        "windows": history or [],
    }
    out = session_dir / "v4_report.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = session_dir / "v4_report.md"
    md.write_text(_format_md(summary), encoding="utf-8")
    return out


def _format_md(summary: Dict[str, Any]) -> str:
    v = summary.get("verdict", "—")
    lines = [
        "# v4 数据质量检测报告",
        "",
        f"- **结论**: {v}",
        f"- **稳定达标**: {'是' if summary.get('achieved_stable') else '否'}",
        f"- **检测时长**: {summary.get('duration_s', '—')} s",
        f"- **好窗/总窗**: {summary.get('pass_windows', '—')}/{summary.get('total_windows', '—')}",
        f"- **通过率**: {float(summary.get('pass_rate', 0)) * 100:.1f}%",
    ]
    if summary.get("time_to_stable_s") is not None:
        lines.append(f"- **达标用时**: {summary['time_to_stable_s']} s")
    if summary.get("median_std_uv_median") is not None:
        lines.append(f"- **中位 std**: {summary['median_std_uv_median']} µV")
    if summary.get("common_mode_ratio_p95") is not None:
        lines.append(f"- **共模比 P95**: {float(summary['common_mode_ratio_p95']) * 100:.1f}%")
    dead = summary.get("chronic_dead_channels") or []
    hot = summary.get("chronic_hot_channels") or []
    if dead:
        lines.append(f"- **反复死通道**: {', '.join(dead)}")
    if hot:
        lines.append(f"- **反复高幅通道**: {', '.join(hot)}")
    lines.extend(["", f"**建议**: {summary.get('recommendation', '')}", ""])
    return "\n".join(lines)
