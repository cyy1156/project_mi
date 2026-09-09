# -*- coding: utf-8 -*-
"""Compare text content of tech-report MD / DOCX / PDF."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from pypdf import PdfReader

OUT = Path(r"D:/MI/资料/初赛材料/01_技术报告/交稿")
MD = OUT / "技术报告_XH-202610_v4.md"
DOCX = OUT / "技术报告_XH-202610.docx"
PDF = OUT / "技术报告_XH-202610.pdf"

W_T = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"


def norm(s: str) -> str:
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = s.replace("｜", "|").replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", "", s)
    # drop markdown/emphasis leftovers
    s = s.replace("**", "").replace("`", "")
    return s


def md_body_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    # strip fenced code lightly; keep content
    lines = []
    in_code = False
    for ln in raw.splitlines():
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            lines.append(ln)
            continue
        # drop pure image lines
        if re.match(r"^!\[[^\]]*\]\([^)]+\)\s*$", ln.strip()):
            continue
        # drop math delimiters but keep latex as text-ish
        ln = ln.replace("$$", " ")
        ln = re.sub(r"\$(?!\$)([^$]+)\$", r"\1", ln)
        # table pipes -> nothing special
        lines.append(ln)
    return "\n".join(lines)


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    # also pull OMML text nodes m:t
    texts = []
    for el in root.iter():
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag == "t" and el.text:
            texts.append(el.text)
        elif tag == "tab":
            texts.append("\t")
        elif tag in ("br", "cr"):
            texts.append("\n")
    return "".join(texts)


def pdf_text(path: Path) -> str:
    r = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in r.pages)


def find_anchors(text: str, anchors: list[str]) -> dict[str, bool]:
    n = norm(text)
    return {a: norm(a) in n for a in anchors}


def section_heads_md(path: Path) -> list[str]:
    heads = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(#{2,4})\s+(.+)$", ln.strip())
        if m:
            heads.append(re.sub(r"\*+", "", m.group(2)).strip())
    return heads


def main():
    md = md_body_text(MD)
    dx = docx_text(DOCX)
    pf = pdf_text(PDF)

    print("=== sizes ===")
    print(f"MD   chars={len(md):6d}  norm={len(norm(md)):6d}  file={MD.stat().st_size}")
    print(f"DOCX chars={len(dx):6d}  norm={len(norm(dx)):6d}  file={DOCX.stat().st_size}")
    print(f"PDF  chars={len(pf):6d}  norm={len(norm(pf)):6d}  file={PDF.stat().st_size}  pages={len(PdfReader(str(PDF)).pages)}")

    anchors = [
        "申报信息",
        "申报学校",
        "上海理工大学",
        "中国智慧工程研究会",
        "面向少样本个性化的智能运动想象脑机交互系统",
        "陈越云",
        "尹钟、陈斌",
        "题目编号：XH-202610",
        "提交日期：2026 年 9 月",
        "关键词",
        "61.88%",
        "51.11%",
        "65.90%",
        "66.30%",
        "47.60%",
        "92.40%",
        "1.11 ms",
        "3.5 s",
        "CausalFuse-8",
        "QuadFold-59",
        "T-Shallow",
        "Shallow-b",
        "第二浅层差异",
        "t0 加权",
        "不同随机种子",
        "12/20",
        "BCI 盲",
        "已超 3 亿",
        "式 (2.1)",
        "式 (2.2)",
        "附录 D",
        "1 引言",
        "2 方法",
        "3 实验结果",
        "4 讨论",
        "5 结论",
    ]

    print("\n=== anchor presence (MD / DOCX / PDF) ===")
    miss = []
    for a in anchors:
        am = norm(a) in norm(md)
        ad = norm(a) in norm(dx)
        ap = norm(a) in norm(pf)
        flag = "OK" if (am and ad and ap) else "DIFF"
        if flag == "DIFF":
            miss.append((a, am, ad, ap))
        print(f"[{flag}] {a}: MD={am} DOCX={ad} PDF={ap}")

    # section heads: md vs presence in docx/pdf
    heads = section_heads_md(MD)
    print(f"\n=== MD section heads ({len(heads)}) vs DOCX/PDF ===")
    head_miss = []
    for h in heads:
        # strip numbering variants
        ad = norm(h) in norm(dx)
        ap = norm(h) in norm(pf)
        if not (ad and ap):
            head_miss.append((h, ad, ap))
            print(f"[MISS] {h}: DOCX={ad} PDF={ap}")
    if not head_miss:
        print("All MD headings found in DOCX and PDF (normalized).")

    # unique long phrases from MD paragraphs (non-table, non-heading)
    phrases = []
    for ln in MD.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("|") or s.startswith("!") or s.startswith("$$") or s.startswith(">"):
            continue
        if s == "---" or s.startswith("```"):
            continue
        s = re.sub(r"\*\*|`", "", s)
        s = re.sub(r"\$(?!\$)([^$]+)\$", r"\1", s)
        if len(norm(s)) >= 40:
            phrases.append(s)
    # sample every Nth + first/last
    sample_idx = sorted(set([0, 1, 2, len(phrases)//4, len(phrases)//2, 3*len(phrases)//4, len(phrases)-2, len(phrases)-1]))
    print(f"\n=== sample MD paragraphs in DOCX/PDF ({len(phrases)} long paras, check {len(sample_idx)}) ===")
    para_miss = []
    for i in sample_idx:
        if i < 0 or i >= len(phrases):
            continue
        p = phrases[i]
        key = norm(p)[:48]  # prefix match more robust for PDF extraction noise
        # for DOCX use longer
        in_dx = norm(p)[:80] in norm(dx) or key in norm(dx)
        in_pf = key in norm(pf)
        if not (in_dx and in_pf):
            para_miss.append((i, p[:60], in_dx, in_pf))
            print(f"[MISS #{i}] DOCX={in_dx} PDF={in_pf} :: {p[:70]}...")
    if not para_miss:
        print("Sampled long paragraphs present in both DOCX and PDF.")

    # DOCX-only / PDF-only cover extras that MD has
    print("\n=== structural extras ===")
    for label, blob in [("DOCX", dx), ("PDF", pf)]:
        for extra in ["目  录", "目录", "（打开后按 F9 更新目录）", "摘  要", "摘要"]:
            print(f"  {label} has '{extra}':", extra.replace(" ", "") in norm(blob) or extra in blob)

    # Numbers that must match counts
    print("\n=== critical number counts ===")
    for num in ["61.88", "51.11", "47.60", "92.40", "66.30", "65.90", "4.7", "1.11", "3.5"]:
        print(f"  {num}: MD={md.count(num)} DOCX={dx.count(num)} PDF={pf.count(num)}")

    print("\n=== SUMMARY ===")
    print(f"anchor mismatches: {len(miss)}")
    print(f"heading mismatches: {len(head_miss)}")
    print(f"paragraph sample mismatches: {len(para_miss)}")
    if miss:
        print("Anchor DIFF detail:")
        for a, am, ad, ap in miss:
            print(f"  - {a}: MD={am} DOCX={ad} PDF={ap}")


if __name__ == "__main__":
    main()
