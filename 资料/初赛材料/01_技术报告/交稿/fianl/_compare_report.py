# -*- coding: utf-8 -*-
"""Compare fianl docx/pdf against 技术报告_XH-202610_v4.md"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

from docx import Document
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DOCX = HERE / "技术报告111.docx"
PDF = HERE / "技术报告1111.pdf"
MD = HERE.parent / "技术报告_XH-202610_v4.md"
OUT = HERE / "_diff_tmp"
OUT.mkdir(exist_ok=True)


def norm(s: str) -> str:
    # drop unpaired surrogates from PDF extract
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = s.replace("－", "-").replace("–", "-")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def md_to_plain(md: str) -> str:
    lines: list[str] = []
    in_code = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.strip().startswith("![") or line.strip() == "---":
            continue
        line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", line)
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^>\s*", "", line)
        line = line.replace("|", " ")
        line = re.sub(r"^[\-\*]\s+", "", line)
        line = re.sub(r"\$\$([^$]+)\$\$", r"\1", line)
        line = re.sub(r"\$([^$]+)\$", r"\1", line)
        lines.append(line)
    return norm("\n".join(lines))


def docx_to_plain(p: Path) -> str:
    doc = Document(str(p))
    parts: list[str] = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return norm("\n".join(parts))


def pdf_to_plain(p: Path) -> str:
    r = PdfReader(str(p))
    parts = [(page.extract_text() or "") for page in r.pages]
    return norm("\n".join(parts))


def sentences(s: str) -> list[str]:
    # split on Chinese/English sentence ends + newlines for headings
    chunks = re.split(r"(?<=[。！？；\n])", s)
    out = []
    for c in chunks:
        c = c.strip()
        if len(collapse_ws(c)) >= 12:
            out.append(c)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    md_raw = MD.read_text(encoding="utf-8")
    md = md_to_plain(md_raw)
    docx = docx_to_plain(DOCX)
    pdf = pdf_to_plain(PDF)

    (OUT / "md.txt").write_text(md, encoding="utf-8")
    (OUT / "docx.txt").write_text(docx, encoding="utf-8")
    (OUT / "pdf.txt").write_text(pdf, encoding="utf-8")

    print("=== sizes ===")
    print(f"md   chars={len(md)}  file={MD.name}")
    print(f"docx chars={len(docx)}  file={DOCX.name}  bytes={DOCX.stat().st_size}")
    print(f"pdf  chars={len(pdf)}  file={PDF.name}  bytes={PDF.stat().st_size}  pages={len(PdfReader(str(PDF)).pages)}")

    md_c, docx_c, pdf_c = collapse_ws(md), collapse_ws(docx), collapse_ws(pdf)
    print("\n=== collapsed equality ===")
    print("docx == md :", docx_c == md_c)
    print("pdf  == md :", pdf_c == md_c)
    print("docx == pdf:", docx_c == pdf_c)
    # similarity ratio on first 80k collapsed
    for name, a, b in [
        ("docx~md", docx_c[:80000], md_c[:80000]),
        ("pdf~md", pdf_c[:80000], md_c[:80000]),
        ("docx~pdf", docx_c[:80000], pdf_c[:80000]),
    ]:
        r = difflib.SequenceMatcher(None, a, b).ratio()
        print(f"ratio {name}: {r:.4f}")

    keys = [
        "61.88",
        "51.11",
        "65.90",
        "66.30",
        "47.60",
        "92.40",
        "37.50",
        "1.11",
        "3.5",
        "QuadFold-59",
        "CausalFuse-8",
        "20 名",
        "54 人",
        "XH-202610",
        "12/20",
        "+4.7",
        "21.1",
        "55.80",
        "41.98",
        "FC3",
        "8–30",
        "8-30",
    ]
    print("\n=== key phrases ===")
    print(f"{'key':12} {'md':5} {'docx':5} {'pdf':5}")
    for k in keys:
        print(f"{k:12} {str(k in md):5} {str(k in docx):5} {str(k in pdf):5}")

    # headings in md
    heads = re.findall(r"^#{1,3}\s+(.+)$", md_raw, re.M)
    print(f"\n=== md headings ({len(heads)}) presence in docx/pdf ===")
    missing_docx, missing_pdf = [], []
    for h in heads:
        h2 = re.sub(r"[*`]", "", h).strip()
        hc = collapse_ws(h2)
        in_d = hc in docx_c or h2 in docx
        in_p = hc in pdf_c or h2 in pdf
        if not in_d:
            missing_docx.append(h2)
        if not in_p:
            missing_pdf.append(h2)
    print("missing in docx:", len(missing_docx))
    for x in missing_docx[:30]:
        print("  -", x)
    print("missing in pdf:", len(missing_pdf))
    for x in missing_pdf[:30]:
        print("  -", x)

    # sentences in md not found in docx/pdf
    md_sents = sentences(md)
    miss_d, miss_p = [], []
    for s in md_sents:
        sc = collapse_ws(s)
        if len(sc) < 20:
            continue
        if sc not in docx_c:
            miss_d.append(s)
        if sc not in pdf_c:
            miss_p.append(s)

    print(f"\n=== md sentences not found in docx ({len(miss_d)}/{len(md_sents)}) ===")
    for s in miss_d[:25]:
        print(" ·", s[:160].replace("\n", " "))
    print(f"\n=== md sentences not found in pdf ({len(miss_p)}/{len(md_sents)}) ===")
    for s in miss_p[:25]:
        print(" ·", s[:160].replace("\n", " "))

    # reverse: unique chunks in docx not in md (sample)
    docx_sents = sentences(docx)
    extra_d = []
    for s in docx_sents:
        sc = collapse_ws(s)
        if len(sc) < 25:
            continue
        if sc not in md_c:
            extra_d.append(s)
    print(f"\n=== docx sentences not found in md ({len(extra_d)}/{len(docx_sents)}) ===")
    for s in extra_d[:25]:
        print(" ·", s[:160].replace("\n", " "))

    pdf_sents = sentences(pdf)
    extra_p = []
    for s in pdf_sents:
        sc = collapse_ws(s)
        if len(sc) < 25:
            continue
        if sc not in md_c:
            extra_p.append(s)
    print(f"\n=== pdf sentences not found in md ({len(extra_p)}/{len(pdf_sents)}) ===")
    for s in extra_p[:25]:
        print(" ·", s[:160].replace("\n", " "))

    # image count
    md_imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_raw)
    print(f"\n=== figures ===")
    print("md image refs:", len(md_imgs))
    for im in md_imgs:
        print(" ", im)

    # docx inline images
    n_blips = 0
    try:
        from docx.opc.constants import RELATIONSHIP_TYPE as RT

        for rel in Document(str(DOCX)).part.rels.values():
            if "image" in rel.reltype:
                n_blips += 1
    except Exception as e:
        print("docx image count fail", e)
    print("docx image relationships:", n_blips)

    # write summary report
    report = OUT / "diff_summary.md"
    lines = [
        "# 技术报告差异摘要（fianl vs v4.md）",
        "",
        f"- md: `{MD.name}`",
        f"- docx: `{DOCX.name}`",
        f"- pdf: `{PDF.name}`",
        "",
        f"- collapsed equality: docx==md={docx_c == md_c}, pdf==md={pdf_c == md_c}, docx==pdf={docx_c == pdf_c}",
        f"- missing headings docx/pdf: {len(missing_docx)}/{len(missing_pdf)}",
        f"- md sentences missing in docx/pdf: {len(miss_d)}/{len(miss_p)}",
        f"- docx/pdf sentences missing in md: {len(extra_d)}/{len(extra_p)}",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\nwrote", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
