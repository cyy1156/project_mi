# -*- coding: utf-8 -*-
from pathlib import Path
import re

md_path = Path(r"D:/MI/资料/初赛材料/01_技术报告/交稿/技术报告_XH-202610_v4.md")
md = md_path.read_text(encoding="utf-8")
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

pdf = PdfReader(r"D:/MI/资料/初赛材料/01_技术报告/交稿/技术报告_XH-202610.pdf")
pdf_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
pdf_n = re.sub(r"\s+", "", pdf_text)
md_n = re.sub(r"\s+", "", md)

anchors = [
    "摘要",
    "20名真人",
    "47.60%",
    "37.50%",
    "12/20",
    "口径与前提",
    "20人登记账",
    "+3.5个百分点",
    "同号比例60.0%",
    "50.0%",
    "板载带通",
    "留一被试",
    "附录D",
    "参考文献",
    "17人冻结",
    "+3.6个百分点",
    "嵌套",
    "交卷",
]
print("PDF pages", len(pdf.pages), "pdf chars", len(pdf_text), "md chars", len(md))
for a in anchors:
    an = re.sub(r"\s+", "", a)
    print(f"{a!r:24s} md={an in md_n} pdf={an in pdf_n}")

heads = re.findall(r"^#{1,3}\s+(.+)$", md, flags=re.M)
missing = []
for h in heads:
    hn = re.sub(r"[\s\*\#]", "", h)
    if hn not in pdf_n:
        missing.append(h)
print("missing headings", len(missing), "/", len(heads))
for h in missing:
    print(" ", h)

# Paragraph count rough
print("\n--- sample PDF page0 ---")
print((pdf.pages[0].extract_text() or "")[:800])
print("\n--- sample PDF around 3.7 ---")
idx = pdf_text.find("3.7")
print(pdf_text[idx : idx + 600] if idx >= 0 else "3.7 NOT FOUND")
