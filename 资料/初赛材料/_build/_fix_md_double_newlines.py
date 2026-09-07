# -*- coding: utf-8 -*-
"""Collapse alternate blank lines introduced by bad rewrite."""
from pathlib import Path

files = [
    Path(r"D:/MI/资料/初赛材料/01_技术报告/技术报告_XH-202610.md"),
    Path(r"D:/MI/资料/初赛材料/01_技术报告/交稿/技术报告_XH-202610_v4.md"),
]


def fix_alt_blanks(text: str) -> str:
    lines = text.splitlines()
    odds = [lines[i] for i in range(1, len(lines), 2)]
    frac = sum(1 for L in odds if L == "") / max(1, len(odds))
    if frac <= 0.85:
        return text
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
    return "\n".join(out) + "\n"


for p in files:
    raw = p.read_text(encoding="utf-8")
    fixed = fix_alt_blanks(raw)
    if fixed != raw:
        p.write_text(fixed, encoding="utf-8", newline="\n")
        print(p.name, "fixed", len(raw.splitlines()), "->", len(fixed.splitlines()))
    else:
        print(p.name, "ok", len(raw.splitlines()))

# ensure sync
src = files[0]
dst = files[1]
dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
print("synced v4")
print("xjh0828", src.read_text(encoding="utf-8").count("xjh0828"))
print("fnz0828", src.read_text(encoding="utf-8").count("fnz0828"))
