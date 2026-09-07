"""Build Exp41 unified real-subject Leave-Next cohort tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(r"D:\MI")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiment_game" / "tools"))

from run_leave_next_e1f_task_ramp import (  # noqa: E402
    SUBJECTS_ALL,
    SUBJECTS_ROOT,
    _list_v3_sessions,
    _ramp_for_subject,
)

SKIP = {
    "_analysis",
    "_backup_old_channel_order_20260829",
    "test",
    "learn_m00",
    "fnz",
    "fnz_1",
    "cyy",
}

OUT_DIR = (
    _REPO
    / "资料"
    / "模型训练"
    / "41_旁路_真人被试LeaveNext_F5全队列统一_openbmi_accpaper"
    / "总结"
)


def prefer_summary(sid: str) -> Path | None:
    ft = SUBJECTS_ROOT / sid / "models" / "ft_runs"
    if not ft.is_dir():
        return None
    all4 = sorted(ft.glob("*leave_next*all4*f5_summary.json"))
    if all4:
        return all4[-1]
    any_ = sorted(ft.glob("*leave_next*f5_summary.json"))
    return any_[-1] if any_ else None


def rows_of(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d, {"ft_scope": None}
    meta = {k: d.get(k) for k in ("ft_scope", "stamp", "subject_id", "protocol") if k in d}
    for k in ("stages", "rows", "summary", "results", "ramp"):
        if isinstance(d.get(k), list):
            return d[k], meta
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "r_stage" in v[0]:
            return v, meta
    return [], meta


def hold_key(heldout: str) -> str:
    for part in str(heldout).replace("+", "_").split("_"):
        p = part.lower()
        if p.startswith("ws") and p[2:].isdigit():
            return p
        if p.startswith("w") and not p.startswith("ws") and p[1:].isdigit():
            return p
    if "w02+w03" in str(heldout) or "w02_w03" in str(heldout):
        return "w02+w03"
    return str(heldout)[-28:]


def lab_cell(by: dict, name: str) -> str:
    b = by.get(name) or {}
    if not b:
        return "-"
    ok, n, acc = b.get("ok"), b.get("n"), b.get("acc")
    pct = 100 * float(acc) if acc is not None else float("nan")
    return f"{ok}/{n}={pct:.1f}%"


def index_sessions(sid: str):
    idx_path = SUBJECTS_ROOT / sid / "index.json"
    if not idx_path.is_file():
        return []
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    return idx.get("sessions") or []


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = sorted(
        p.name for p in SUBJECTS_ROOT.iterdir() if p.is_dir() and p.name not in SKIP
    )

    # --- structure markdown ---
    struct_lines = [
        "# 实验 41 · 真人被试结构总表（session / 爬坡）",
        "",
        "> 生成：脚本扫描 `experiment_game/data/subjects` + `run_leave_next_e1f_task_ramp.py`",
        "> v4 / record_excluded / 硬排除半场仍踢出；**电极 CZ 饱和（ft_eligible=false）不再排除**（2026-09-04）",
        "",
        "## 被试清单与协议位",
        "",
        "| 被试 | SUBJECTS_ALL | 最新 summary | scope |",
        "|------|--------------|--------------|-------|",
    ]

    cohort_end = [
        "",
        "## 末档总表（优先最新 all4 Leave-Next）",
        "",
        "| 被试 | nR | 末档 hold | 三分类窗smooth | 三分类窗raw | FT F5 MI | E1f零样本 MI | 总分 | PASS | JSON |",
        "|------|----|-----------|----------------|-------------|----------|--------------|------|------|------|",
    ]

    detail_parts = ["", "## 分被试分档表", ""]

    session_parts = ["", "## 分被试 session 结构", ""]

    for sid in subjects:
        summ = prefer_summary(sid)
        scope = "-"
        stamp = "-"
        if summ:
            rows, meta = rows_of(summ)
            scope = "all4" if "all4" in summ.name else (meta.get("ft_scope") or "e1f_task/other")
            stamp = summ.name
        else:
            rows = []
        in_all = sid in SUBJECTS_ALL
        struct_lines.append(
            f"| {sid} | {in_all} | `{stamp}` | {scope} |"
        )

        # sessions
        session_parts.append(f"### {sid}")
        session_parts.append("")
        session_parts.append(
            "| dir | phase | ft_eligible | excluded | electrode | primary_acc | window_acc |"
        )
        session_parts.append(
            "|-----|-------|-------------|----------|-----------|-------------|------------|"
        )
        for s in index_sessions(sid):
            session_parts.append(
                f"| `{s.get('dir')}` | {s.get('phase_mode')} | {s.get('ft_eligible')} | "
                f"{s.get('record_excluded')} | {s.get('electrode_ok')} | "
                f"{s.get('primary_acc')} | {s.get('window_acc')} |"
            )
        by = _list_v3_sessions(sid) if in_all or True else {}
        try:
            by = _list_v3_sessions(sid)
        except Exception:
            by = {}
        session_parts.append("")
        session_parts.append(f"- Leave-Next 可用键：`{sorted(by.keys())}`")
        if in_all:
            try:
                ramp = _ramp_for_subject(sid, by)
                session_parts.append(f"- 爬坡档数：**{len(ramp)}**")
                for i, (tr, hold, rep) in enumerate(ramp, 1):
                    session_parts.append(
                        f"  - R{i}: train=`{tr}` → hold=`{hold}` · replay={rep}"
                    )
            except Exception as e:
                session_parts.append(f"- 爬坡解析失败：{e}")
        else:
            session_parts.append("- **未在 SUBJECTS_ALL**（需加脚本才可自动爬坡）")
        session_parts.append("")

        if not rows:
            cohort_end.append(f"| {sid} | 0 | — | — | — | — | — | — | — |")
            continue

        # detail table
        detail_parts.append(f"### {sid}")
        detail_parts.append("")
        detail_parts.append(f"- JSON：`experiment_game/data/subjects/{sid}/models/ft_runs/{stamp}`")
        detail_parts.append(f"- scope 推断：**{scope}**")
        detail_parts.append("")
        detail_parts.append(
            "| R | hold | 三分类窗smooth | 三分类窗raw | FT F5 MI | E1f零样本 MI | 总分 FT | Left | Right | Rest | PASS |"
        )
        detail_parts.append(
            "|---|------|----------------|-------------|----------|--------------|---------|------|-------|------|------|"
        )
        for r in rows:
            f5 = r.get("f5_ft") or {}
            be = r.get("f5_base_e1f") or {}
            by_lab = f5.get("by_label") or {}
            win = r.get("heldout_acc_smooth", r.get("heldout_acc"))
            win_raw = r.get("heldout_acc_raw", f5.get("window_acc"))
            mi = f5.get("mi_acc")
            e1f = be.get("mi_acc")
            score, smax = f5.get("score"), f5.get("score_max")
            win_s = f"{float(win):.3f}" if isinstance(win, (int, float)) else str(win)
            win_r = (
                f"{float(win_raw):.3f}"
                if isinstance(win_raw, (int, float))
                else str(win_raw)
            )
            mi_s = f"{100 * float(mi):.1f}%" if isinstance(mi, (int, float)) else str(mi)
            e1f_s = (
                f"{100 * float(e1f):.1f}%" if isinstance(e1f, (int, float)) else "-"
            )
            detail_parts.append(
                f"| {r.get('r_stage')} | {hold_key(r.get('heldout', ''))} | {win_s} | "
                f"{win_r} | {mi_s} | {e1f_s} | {score}/{smax} | "
                f"{lab_cell(by_lab, 'Left')} | {lab_cell(by_lab, 'Right')} | "
                f"{lab_cell(by_lab, 'Rest')} | {r.get('release_pass')} |"
            )
        detail_parts.append("")

        last = rows[-1]
        f5 = last.get("f5_ft") or {}
        be = last.get("f5_base_e1f") or {}
        win = last.get("heldout_acc_smooth", last.get("heldout_acc"))
        win_raw = last.get("heldout_acc_raw", f5.get("window_acc"))
        mi = f5.get("mi_acc")
        e1f = be.get("mi_acc")
        score, smax = f5.get("score"), f5.get("score_max")
        win_s = f"{float(win):.3f}" if isinstance(win, (int, float)) else str(win)
        win_r = (
            f"{float(win_raw):.3f}"
            if isinstance(win_raw, (int, float))
            else str(win_raw)
        )
        mi_s = f"{100 * float(mi):.1f}%" if isinstance(mi, (int, float)) else str(mi)
        e1f_s = f"{100 * float(e1f):.1f}%" if isinstance(e1f, (int, float)) else "-"
        cohort_end.append(
            f"| {sid} | {len(rows)} | {hold_key(last.get('heldout', ''))} | {win_s} | "
            f"{win_r} | {mi_s} | {e1f_s} | {score}/{smax} | "
            f"{last.get('release_pass')} | `{stamp}` |"
        )

    # sort cohort by MI desc for a ranked view - rebuild from lines later in registry
    text = "\n".join(struct_lines + cohort_end + session_parts + detail_parts) + "\n"
    out = OUT_DIR / "_generated_cohort.md"
    out.write_text(text, encoding="utf-8")
    # also dump json index
    index = []
    for sid in subjects:
        summ = prefer_summary(sid)
        by = _list_v3_sessions(sid)
        item = {
            "subject_id": sid,
            "in_subjects_all": sid in SUBJECTS_ALL,
            "include_ft_ineligible": False,  # 已废弃：电极饱和不再排除
            "leave_next_keys": sorted(by.keys()),
            "summary": summ.name if summ else None,
            "summary_path": str(summ) if summ else None,
        }
        if summ:
            rows, _ = rows_of(summ)
            item["n_rounds"] = len(rows)
            if rows:
                last = rows[-1]
                f5 = last.get("f5_ft") or {}
                be = last.get("f5_base_e1f") or {}
                item["last"] = {
                    "r": last.get("r_stage"),
                    "hold": hold_key(last.get("heldout", "")),
                    "win": last.get("heldout_acc_smooth", last.get("heldout_acc")),
                    "win_smooth": last.get(
                        "heldout_acc_smooth", last.get("heldout_acc")
                    ),
                    "win_raw": last.get(
                        "heldout_acc_raw", f5.get("window_acc")
                    ),
                    "f5_win3": f5.get("window_acc"),
                    "mi": f5.get("mi_acc"),
                    "e1f_mi": be.get("mi_acc"),
                    "pass": last.get("release_pass"),
                    "score": f5.get("score"),
                    "score_max": f5.get("score_max"),
                }
        index.append(item)
    (OUT_DIR / "cohort_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", out)
    print("subjects", len(subjects))
    # chain: refresh human-facing registry
    try:
        from _build_exp41_registry import main as build_registry

        build_registry()
    except Exception as exc:
        print("registry build skipped:", exc)


if __name__ == "__main__":
    main()
