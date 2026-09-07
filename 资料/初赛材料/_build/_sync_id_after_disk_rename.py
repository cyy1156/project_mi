# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

OLD = "fnz0828"
NEW = "xjh0828"


def replace_file(path: Path, old: str, new: str) -> bool:
    raw = path.read_text(encoding="utf-8")
    if old not in raw:
        return False
    path.write_text(raw.replace(old, new), encoding="utf-8")
    return True


def main() -> None:
    # 1) live cohort_map
    p = Path(
        r"D:/MI/code/train_lab/src/step/5070_exp42_confound_collapse_accpaper/cohort_map.py"
    )
    t = p.read_text(encoding="utf-8")
    t2 = t.replace(
        '# 2026-09-05 口径修订：fnz0828 与 fnz0830 按登记行各自为独立个体（n_people=17，\n'
        '# 与 v4 报告 §3.6 对齐）；同日重复壳 fnz / fnz_1 仍归入 fnz0828。\n'
        "MERGE_RULES: Dict[str, List[str]] = {\n"
        '    "fnz0828": ["fnz", "fnz0828", "fnz_1"],\n',
        '# 2026-09-05 口径修订：xjh0828 与 fnz0830 按登记行各自为独立个体；\n'
        "# 同日重复壳 fnz / fnz_1 仍归入 xjh0828；磁盘主键已改为 xjh0828（fnz0828 仅作历史别名）。\n"
        "MERGE_RULES: Dict[str, List[str]] = {\n"
        '    "xjh0828": ["fnz", "fnz0828", "xjh0828", "fnz_1"],\n',
    )
    if t2 == t:
        raise SystemExit("cohort_map block not found")
    p.write_text(t2, encoding="utf-8")
    print("updated cohort_map")

    # 2) related live modules
    pairs = [
        (
            Path(
                r"D:/MI/code/train_lab/src/step/5070_exp42_confound_collapse_accpaper/io_sessions.py"
            ),
            'if member_id == "fnz0828" and d.name.endswith("_152231"):',
            'if member_id in ("xjh0828", "fnz0828") and d.name.endswith("_152231"):',
        ),
        (
            Path(
                r"D:/MI/code/train_lab/src/step/5070_exp42_confound_collapse_accpaper/feat_anchor_check.py"
            ),
            'CHECK_PEOPLE = ["syj0828", "fnz0828", "zyj0902"]',
            'CHECK_PEOPLE = ["syj0828", "xjh0828", "zyj0902"]',
        ),
    ]
    for path, a, b in pairs:
        tt = path.read_text(encoding="utf-8")
        if a not in tt:
            print("MISS", path.name)
            continue
        path.write_text(tt.replace(a, b), encoding="utf-8")
        print("ok", path.name)

    sp = Path(
        r"D:/MI/code/train_lab/src/step/5070_exp42_confound_collapse_accpaper/summary_42.py"
    )
    if replace_file(sp, "fnz0828/fnz0830", "xjh0828/fnz0830"):
        print("summary_42")

    # 3) analysis_42 artifacts
    ana = Path(
        r"D:/MI/资料/模型训练/42_旁路_真人队列混杂分解与会话特征坍塌诊断_accpaper"
    )
    n = 0
    for f in ana.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".json", ".md", ".csv", ".txt", ".py", ".yaml", ".yml"}:
            continue
        if replace_file(f, OLD, NEW):
            n += 1
    print("analysis replaced", n)

    # 4) experiment_game tools (operational)
    tools_root = Path(r"D:/MI/experiment_game/tools")
    n2 = 0
    for f in tools_root.rglob("*.py"):
        raw = f.read_text(encoding="utf-8")
        if OLD not in raw:
            continue
        nt = raw.replace(OLD, NEW)
        # restore dual-match for skip-v4-ws01 where useful
        # (session dirs now start with xjh0828_..., but keep alias for old logs)
        f.write_text(nt, encoding="utf-8")
        n2 += 1
        print("tool", f.relative_to(tools_root))
    print("tools updated", n2)

    # 5) patch equality checks back to dual where session skip used
    dual_targets = [
        Path(r"D:/MI/experiment_game/tools/run_leave_next_e1f_task_ramp.py"),
        Path(r"D:/MI/experiment_game/tools/run_real_subject_all4_vs_so.py"),
        Path(r"D:/MI/code/train_lab/src/step/5070_exp42_confound_collapse_accpaper/io_sessions.py"),
    ]
    for f in dual_targets:
        if not f.exists():
            continue
        t = f.read_text(encoding="utf-8")
        t2 = t
        t2 = t2.replace(
            'subject_id == "xjh0828"',
            'subject_id in ("xjh0828", "fnz0828")',
        )
        t2 = t2.replace(
            'member_id == "xjh0828"',
            'member_id in ("xjh0828", "fnz0828")',
        )
        t2 = t2.replace(
            'sid == "xjh0828"',
            'sid in ("xjh0828", "fnz0828")',
        )
        # avoid double-wrapping
        t2 = t2.replace(
            'subject_id in ("xjh0828", "fnz0828") in ("xjh0828", "fnz0828")',
            'subject_id in ("xjh0828", "fnz0828")',
        )
        t2 = t2.replace(
            'member_id in ("xjh0828", "fnz0828") in ("xjh0828", "fnz0828")',
            'member_id in ("xjh0828", "fnz0828")',
        )
        if t2 != t:
            f.write_text(t2, encoding="utf-8")
            print("dual", f.name)

    # 6) make_result_figures comment
    fig = Path(r"D:/MI/资料/初赛材料/01_技术报告/make_result_figures.py")
    ft = fig.read_text(encoding="utf-8")
    ft2 = ft.replace(
        "        # 磁盘目录可能仍为 fnz0828；对外报告统一显示 xjh0828\n"
        '        sid = str(d.get("subject_id") or p.parent.parent.parent.name)\n'
        '        sid = {"fnz0828": "xjh0828"}.get(sid, sid)\n',
        "        # 兼容历史 summary 中的 fnz0828 主键\n"
        '        sid = str(d.get("subject_id") or p.parent.parent.parent.name)\n'
        '        sid = {"fnz0828": "xjh0828"}.get(sid, sid)\n',
    )
    if ft2 != ft:
        fig.write_text(ft2, encoding="utf-8")
        print("figures comment")

    print("DONE")


if __name__ == "__main__":
    main()
