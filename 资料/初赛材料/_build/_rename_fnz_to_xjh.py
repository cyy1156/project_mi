# -*- coding: utf-8 -*-
from pathlib import Path

p = Path(r"D:/MI/资料/初赛材料/01_技术报告/make_result_figures.py")
lines = p.read_text(encoding="utf-8").splitlines()
odds = [lines[i] for i in range(1, len(lines), 2)]
frac = sum(1 for L in odds if L == "") / max(1, len(odds))
print("odd blank frac", round(frac, 3), "n", len(lines))
if frac > 0.85:
    fixed = [lines[i] for i in range(0, len(lines), 2)]
    out = []
    blank_run = 0
    for L in fixed:
        if L == "":
            blank_run += 1
            if blank_run <= 1:
                out.append(L)
        else:
            blank_run = 0
            out.append(L)
    p.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("fixed to", len(out), "lines")

text = p.read_text(encoding="utf-8")
old = '        sid = str(d.get("subject_id") or p.parent.parent.parent.name)\n'
new = (
    '        # 磁盘目录可能仍为 fnz0828；对外报告统一显示 xjh0828\n'
    '        sid = str(d.get("subject_id") or p.parent.parent.parent.name)\n'
    '        sid = {"fnz0828": "xjh0828"}.get(sid, sid)\n'
)
if 'sid = {"fnz0828": "xjh0828"}.get(sid, sid)' in text:
    print("alias already present")
elif old not in text:
    raise SystemExit("anchor not found for sid inject")
else:
    text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8", newline="\n")
    print("alias injected")

cm = Path(
    r"D:/MI/资料/初赛材料/04_代码包/交稿/code/train_lab/src/step/"
    r"5070_exp42_confound_collapse_accpaper/cohort_map.py"
)
ct = cm.read_text(encoding="utf-8")
ct2 = ct.replace(
    '"xjh0828": ["fnz", "xjh0828", "fnz_1"]',
    '"xjh0828": ["fnz", "fnz0828", "xjh0828", "fnz_1"]',
)
if ct2 != ct:
    cm.write_text(ct2, encoding="utf-8")
    print("cohort_map disk alias restored")
else:
    print("cohort_map unchanged")

# sync root report md already done; ensure v4 == source
src = Path(r"D:/MI/资料/初赛材料/01_技术报告/技术报告_XH-202610.md")
v4 = Path(r"D:/MI/资料/初赛材料/01_技术报告/交稿/技术报告_XH-202610_v4.md")
v4.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
print("synced v4 md")
print("fnz0828 left in report", "fnz0828" in src.read_text(encoding="utf-8"))
print("xjh0828 in report", "xjh0828" in src.read_text(encoding="utf-8"))
