"""汇总 Leave-Next F5 结果（0828/0830 队列）→ markdown。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[2]
SUBJECTS_ROOT = _REPO / "experiment_game" / "data" / "subjects"
OUT_ROOT = SUBJECTS_ROOT / "_analysis"


def _latest_summary(subject_id: str, *, stamp_prefix: str = "") -> Path | None:
    ft = SUBJECTS_ROOT / subject_id / "models" / "ft_runs"
    if not ft.is_dir():
        return None
    cands = sorted(
        ft.glob(f"{stamp_prefix}*{subject_id}*leave_next*f5_summary.json")
        if stamp_prefix
        else ft.glob(f"*{subject_id}*leave_next*f5_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        cands = sorted(
            ft.glob("*e1f_task_leave_next_f5_summary.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    return cands[0] if cands else None


def _f5_mi(pack: Dict[str, Any] | None) -> str:
    if not pack:
        return "—"
    bl = pack.get("by_label") or {}
    left = bl.get("Left") or {}
    right = bl.get("Right") or {}
    rest = bl.get("Rest") or {}
    n_mi = int(left.get("n") or 0) + int(right.get("n") or 0)
    ok_mi = int(left.get("ok") or 0) + int(right.get("ok") or 0)
    n_r = int(rest.get("n") or 0)
    ok_r = int(rest.get("ok") or 0)
    mi = f"{ok_mi}/{n_mi}" if n_mi else "—"
    rs = f"{ok_r}/{n_r}" if n_r else "0/0"
    score = pack.get("score")
    smax = pack.get("score_max")
    sc = f"{score:.1f}/{smax:.1f}" if score is not None and smax is not None else "—"
    return f"MI {mi} Rest {rs} score {sc}"


def main(subjects: List[str], stamp: str) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Leave-Next F5 复验（Cue前静息=Rest 切窗）· {stamp}",
        "",
        "口径：shallow three Leave-Next；F5=因果平滑 lookback=2 + 多数票；",
        "FT 切窗已含 trial_table `t_rest_*`（Rest/label=0）。",
        "",
    ]
    rows_all: List[Dict[str, Any]] = []
    for sid in subjects:
        sp = _latest_summary(sid)
        lines.append(f"## {sid}")
        if sp is None:
            lines.append("- 无 summary")
            lines.append("")
            continue
        payload = json.loads(sp.read_text(encoding="utf-8"))
        lines.append(f"- summary: `{sp}`")
        lines.append("")
        lines.append("| R | train→eval | replay | win heldout | gate | F5 FT | F5 base3 | pred |")
        lines.append("|---|------------|--------|-------------|------|-------|----------|------|")
        for i, row in enumerate(payload.get("rows") or [], start=1):
            train = "+".join(row.get("train") or [])
            hold = row.get("heldout") or "?"
            rep = "on" if row.get("use_replay") else "off"
            wh = row.get("heldout_acc")
            wh_s = f"{wh:.3f}" if isinstance(wh, (int, float)) else "—"
            gate = "PASS" if row.get("release_pass") else "FAIL"
            pred = row.get("pred_labels") or row.get("checks") or {}
            if isinstance(pred, dict) and "Rest" not in str(pred):
                # release may store under checks only
                pass
            # try read release from row
            pred_s = ""
            rg = row.get("release_gate") or {}
            pl = rg.get("pred_labels") or row.get("pred_labels") or {}
            if pl:
                pred_s = ", ".join(f"{k}:{v}" for k, v in pl.items())
            else:
                pred_s = "—"
            lines.append(
                f"| R{i} | {train}→{hold} | {rep} | {wh_s} | {gate} | "
                f"{_f5_mi(row.get('f5_ft'))} | {_f5_mi(row.get('f5_base_three'))} | {pred_s} |"
            )
            rows_all.append({"subject": sid, **row})
        lines.append("")

    out = OUT_ROOT / f"leave_next_f5_restfix_{stamp}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    js = OUT_ROOT / f"leave_next_f5_restfix_{stamp}.json"
    js.write_text(
        json.dumps(
            {"generated_at": datetime.now().isoformat(timespec="seconds"), "rows": rows_all},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subs = sys.argv[1:] or [
        "syj0828",
        "fnz0828",
        "cyy0830",
        "fnz0830",
        "wzr0830",
        "xj0830",
    ]
    main(subs, stamp)
