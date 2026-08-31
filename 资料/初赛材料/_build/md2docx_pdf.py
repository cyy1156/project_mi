#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""技术报告 markdown → DOCX（python-docx）与 PDF（reportlab）双渲染。

排版遵循 docx skill Profile A（正式报告）：黑体标题、宋体正文 12pt、
1.3 倍行距、正文首行缩进 2 字符、表头跨页重复、行禁拆。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "01_技术报告" / "技术报告_XH-202610.md"
DOCX_OUT = ROOT / "01_技术报告" / "技术报告_XH-202610.docx"
PDF_OUT = ROOT / "01_技术报告" / "技术报告_XH-202610.pdf"

# ---------------------------------------------------------------- 解析
Block = tuple  # (kind, payload)


def parse(md_text: str):
    lines = md_text.splitlines()
    blocks: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", "\n".join(buf)))
            continue
        if s.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        if s.startswith("### "):
            blocks.append(("h3", s[4:].strip()))
        elif s.startswith("## "):
            blocks.append(("h2", s[3:].strip()))
        elif s.startswith("# "):
            blocks.append(("h1", s[2:].strip()))
        elif s == "---":
            blocks.append(("hr", None))
        elif s.startswith("> "):
            blocks.append(("quote", s[2:].strip()))
        elif re.match(r"^\d+\.\s", s):
            blocks.append(("li_num", re.sub(r"^\d+\.\s+", "", s)))
        elif s.startswith("- "):
            blocks.append(("li_bul", s[2:].strip()))
        else:
            blocks.append(("p", s))
        i += 1
    return blocks


# ---------------------------------------------------------------- DOCX
def build_docx(blocks, out: Path):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, Twips

    doc = Document()

    # 页面 A4 + 页边距
    for sec in doc.sections:
        sec.page_width, sec.page_height = Twips(11906), Twips(16838)
        sec.top_margin = sec.bottom_margin = Twips(1440)
        sec.left_margin = sec.right_margin = Twips(1600)

    def set_fonts(run, cn="宋体", en="Times New Roman", size=12, bold=False, color="000000"):
        run.font.name = en
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = __import__("docx.shared", fromlist=["RGBColor"]).RGBColor.from_string(color)
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:ascii"), en)
        rfonts.set(qn("w:hAnsi"), en)
        rfonts.set(qn("w:eastAsia"), cn)

    INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")

    def add_inline(p, text, size=12, bold_all=False, cn="宋体"):
        for part in INLINE.split(text):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                r = p.add_run(part[2:-2])
                set_fonts(r, cn=cn, size=size, bold=True)
            elif part.startswith("`") and part.endswith("`"):
                r = p.add_run(part[1:-1])
                set_fonts(r, cn="Consolas", en="Consolas", size=size - 1.5)
            else:
                r = p.add_run(part)
                set_fonts(r, cn=cn, size=size, bold=bold_all)

    def para(text, size=12, indent=480, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=6, cn="宋体"):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.3
        pf.space_before, pf.space_after = Pt(before), Pt(after)
        pf.alignment = align
        if indent:
            pf.first_line_indent = Twips(indent)
        add_inline(p, text, size=size, cn=cn)
        return p

    def heading(text, level):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.3
        pf.space_before, pf.space_after = Pt(14 if level == 1 else 10), Pt(8)
        if level == 1:
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hmap = {1: ("Heading 1", 16), 2: ("Heading 2", 15), 3: ("Heading 3", 14)}
        style, size = hmap[level]
        p.style = doc.styles[style]
        add_inline(p, text, size=size, bold_all=True, cn="黑体")
        for r in p.runs:
            set_fonts(r, cn="黑体", size=size, bold=True)
        return p

    def shade(el, fill):
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), fill)
        el.append(shd)

    def table(rows):
        ncol = max(len(r) for r in rows)
        t = doc.add_table(rows=len(rows), cols=ncol)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        # 单元格边距
        tblpr = t._tbl.tblPr
        mar = OxmlElement("w:tblCellMar")
        for side, w in (("top", 40), ("bottom", 40), ("left", 80), ("right", 80)):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), str(w))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        tblpr.append(mar)
        fsize = 10.5 if ncol <= 6 else 9
        for ri, row in enumerate(rows):
            trpr = t.rows[ri]._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            trpr.append(cant)
            if ri == 0:
                th = OxmlElement("w:tblHeader")
                trpr.append(th)
            for ci in range(ncol):
                cell = t.cell(ri, ci)
                text = row[ci] if ci < len(row) else ""
                p = cell.paragraphs[0]
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else WD_ALIGN_PARAGRAPH.LEFT
                add_inline(p, text, size=fsize, bold_all=(ri == 0))
                if ri == 0:
                    shade(cell._tc.get_or_add_tcPr(), "D9E2F3")

    # ---- 封面（简单标题页）----
    title = next(v for k, v in blocks if k == "h1")
    meta = next(v for k, v in blocks if k == "p" and v.startswith("**题目编号"))
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(200)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    set_fonts(r, cn="黑体", size=24, bold=True)
    doc.add_paragraph()
    for mtext in ("基于运动想象的脑-机交互算法研究与系统实现", meta.strip("*")):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(mtext)
        set_fonts(r, cn="宋体", size=14)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---- 摘要 ----
    heading("摘要", 2)
    for k, v in blocks:
        if k == "quote":
            para(v, size=12, indent=480)
    para("**关键词：**运动想象；脑机接口；少样本个性化适配；迁移学习；异步交互协议", indent=480)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---- 目录（域，需在 Word 中更新）----
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("目  录")
    set_fonts(r, cn="黑体", size=16, bold=True)
    fldp = doc.add_paragraph()
    r1 = fldp.add_run()
    fc1 = OxmlElement("w:fldChar"); fc1.set(qn("w:fldCharType"), "begin"); fc1.set(qn("w:dirty"), "true")
    r1._r.append(fc1)
    r2 = fldp.add_run()
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
    it.text = r' TOC \o "1-3" \h \z \u '
    r2._r.append(it)
    r3 = fldp.add_run()
    fc2 = OxmlElement("w:fldChar"); fc2.set(qn("w:fldCharType"), "separate")
    r3._r.append(fc2)
    r4 = fldp.add_run("（打开后按 F9 更新目录）")
    set_fonts(r4, cn="宋体", size=10.5)
    r5 = fldp.add_run()
    fc3 = OxmlElement("w:fldChar"); fc3.set(qn("w:fldCharType"), "end")
    r5._r.append(fc3)
    hint = doc.add_paragraph()
    hint.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = hint.add_run("提示：打开文档后按 Ctrl+A → F9（或右键目录 → 更新域）刷新目录页码。")
    set_fonts(r, cn="宋体", size=9)
    r.font.color.rgb = __import__("docx.shared", fromlist=["RGBColor"]).RGBColor.from_string("808080")
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---- 正文 ----
    seen_body = False
    for k, v in blocks:
        if k == "hr" and not seen_body:
            seen_body = True
            continue
        if not seen_body:
            continue
        if k == "hr":
            continue
        if k == "h1":
            continue  # 封面已用
        if k == "h2":
            heading(v, 1)
        elif k == "h3":
            heading(v, 2)
        elif k == "p":
            if v.startswith("**题目编号"):
                continue
            para(v)
        elif k == "quote":
            para(v, size=10.5, indent=480)
        elif k == "code":
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.line_spacing = 1.1
            pf.left_indent = Twips(480)
            shade(p._p.get_or_add_pPr(), "F2F2F2")
            for seg in v.split("\n"):
                r = p.add_run(seg)
                set_fonts(r, cn="宋体", en="Consolas", size=9)
                r.add_break()
        elif k == "table":
            table(v)
            doc.add_paragraph().paragraph_format.space_after = Pt(3)
        elif k == "li_num":
            p = para(v, indent=0)
            p.paragraph_format.left_indent = Twips(480)
            add_inline(p, "· ", size=12)
            p.runs.clear() if False else None
        elif k == "li_bul":
            p = para(v, indent=0)
            p.paragraph_format.left_indent = Twips(480)

    # 页脚页码
    for sec in doc.sections:
        fp = sec.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), "PAGE \\* arabic \\* MERGEFORMAT")
        fp._p.append(fld)

    doc.save(out)
    print("docx saved:", out)


# ---------------------------------------------------------------- PDF
def build_pdf(blocks, out: Path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate, Paragraph, Preformatted,
        Spacer, Table, TableStyle, PageBreak,
    )
    from reportlab.platypus.tableofcontents import TableOfContents

    FONT_DIR = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("SimSun", str(FONT_DIR / "simsun.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("SimHei", str(FONT_DIR / "simhei.ttf")))
    pdfmetrics.registerFontFamily("SimSun", normal="SimSun", bold="SimHei", italic="SimSun", boldItalic="SimHei")

    BLACK = colors.HexColor("#000000")
    HEAD_BG = colors.HexColor("#D9E2F3")
    CODE_BG = colors.HexColor("#F2F2F2")
    GRAY = colors.HexColor("#606060")

    def st(name, **kw):
        base = dict(fontName="SimSun", fontSize=12, leading=12 * 1.3, textColor=BLACK,
                    alignment=TA_JUSTIFY, spaceAfter=6, wordWrap="CJK")
        base.update(kw)
        return ParagraphStyle(name, **base)

    S = {
        "h1": st("h1", fontName="SimHei", fontSize=16, leading=16 * 1.3, alignment=TA_CENTER,
                 spaceBefore=14, spaceAfter=8, keepWithNext=1),
        "h2": st("h2", fontName="SimHei", fontSize=15, leading=15 * 1.3, spaceBefore=10, spaceAfter=8,
                 alignment=TA_LEFT, keepWithNext=1),
        "h3": st("h3", fontName="SimHei", fontSize=14, leading=14 * 1.3, spaceBefore=10, spaceAfter=6,
                 alignment=TA_LEFT, keepWithNext=1),
        "body": st("body", firstLineIndent=24),
        "quote": st("quote", fontSize=10.5, leading=10.5 * 1.3, firstLineIndent=21, textColor=BLACK),
        "code": st("code", fontName="Courier", fontSize=9, leading=11, alignment=TA_LEFT),
        "code_cjk": st("code_cjk", fontName="SimSun", fontSize=9.5, leading=13, alignment=TA_LEFT),
        "li": st("li", leftIndent=24, firstLineIndent=0),
        "cell": st("cell", fontSize=9, leading=11.5, alignment=TA_LEFT, spaceAfter=0),
        "cellh": st("cellh", fontName="SimHei", fontSize=9, leading=11.5, alignment=TA_CENTER, spaceAfter=0),
        "cover_title": st("cover_title", fontName="SimHei", fontSize=24, leading=34, alignment=TA_CENTER),
        "cover_meta": st("cover_meta", fontSize=14, leading=20, alignment=TA_CENTER),
    }

    def esc(t):
        return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                 .replace("\u2212", "-").replace("\u2010", "-"))

    def inline(t):
        t = esc(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)

        def _code(m):
            s = m.group(1)
            if any("\u4e00" <= ch <= "\u9fff" for ch in s):
                return s  # 含中文的路径名用正文字体，Courier 缺 CJK 字形
            return f'<font face="Courier" size="10">{s}</font>'

        t = re.sub(r"`([^`]+)`", _code, t)
        return t

    story: list = []
    toc = TableOfContents()
    toc.levelStyles = [
        st("toc1", fontName="SimHei", fontSize=12, leading=18, alignment=TA_LEFT, leftIndent=6),
        st("toc2", fontSize=10.5, leading=15, alignment=TA_LEFT, leftIndent=24),
    ]

    # 封面
    title = next(v for k, v in blocks if k == "h1")
    meta = next(v for k, v in blocks if k == "p" and v.startswith("**题目编号")).strip("*")
    meta_parts = meta.rsplit("｜", 1)
    meta_lines = [p_.strip() for p_ in meta_parts if p_.strip()] if len(meta_parts) == 2 else [meta]
    story += [Spacer(1, 55 * mm), Paragraph(esc(title), S["cover_title"]), Spacer(1, 8 * mm)]
    story.append(Paragraph(esc("基于运动想象的脑-机交互算法研究与系统实现"), S["cover_meta"]))
    story.append(Spacer(1, 3 * mm))
    for ml in meta_lines:
        story.append(Paragraph(esc(ml), S["cover_meta"]))
    story.append(PageBreak())

    # 摘要
    story.append(Paragraph("摘  要", S["h2"]))
    for k, v in blocks:
        if k == "quote":
            story.append(Paragraph(inline(v), S["body"]))
    story.append(Paragraph(inline("**关键词：**运动想象；脑机接口；少样本个性化适配；迁移学习；异步交互协议"), S["body"]))
    story.append(PageBreak())

    # 目录
    story.append(Paragraph("目  录", S["h1"]))
    story.append(Spacer(1, 4 * mm))
    story.append(toc)
    story.append(PageBreak())

    # 正文（带 TOC 收集）：跳过标题/摘要区，从第一个分隔线后开始
    seen_body = False

    class Doc(BaseDocTemplate):
        def afterFlowable(self, fl):
            if isinstance(fl, Paragraph) and getattr(fl, "_toc_entry", None):
                level, text = fl._toc_entry
                self.notify("TOCEntry", (level, text, self.page))

    def h(text, level):
        p = Paragraph(inline(text), S[{1: "h1", 2: "h2", 3: "h3"}[level]])
        p._toc_entry = (level - 1, re.sub(r"\*\*|`", "", text))
        return p

    for k, v in blocks:
        if k == "hr" and not seen_body:
            seen_body = True
            continue
        if not seen_body:
            continue
        if k == "hr":
            continue
        if k == "h1":
            continue
        if k == "h2":
            story.append(h(v, 1))
        elif k == "h3":
            story.append(h(v, 2))
        elif k == "p":
            if v.startswith("**题目编号"):
                continue
            story.append(Paragraph(inline(v), S["body"]))
        elif k == "quote":
            story.append(Paragraph(inline(v), S["quote"]))
        elif k == "code":
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in v)
            style = S["code_cjk"] if has_cjk else S["code"]
            story.append(Preformatted(v, style, maxLineLength=90))
            story.append(Spacer(1, 3 * mm))
        elif k == "table":
            ncol = max(len(r) for r in v)
            data = []
            for ri, row in enumerate(v):
                data.append([Paragraph(inline(row[ci] if ci < len(row) else ""), S["cellh" if ri == 0 else "cell"])
                             for ci in range(ncol)])
            t = Table(data, colWidths=None, repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#909090")),
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F5")]),
            ]))
            story += [t, Spacer(1, 3 * mm)]
        elif k in ("li_num", "li_bul"):
            story.append(Paragraph("• " + inline(v), S["li"]))

    def on_page(canv, doc_):
        canv.saveState()
        canv.setFont("SimSun", 9)
        canv.drawCentredString(A4[0] / 2, 12 * mm, str(canv.getPageNumber()))
        canv.restoreState()

    doc_ = Doc(str(out), pagesize=A4,
               leftMargin=25 * mm, rightMargin=25 * mm, topMargin=22 * mm, bottomMargin=22 * mm,
               title="基于运动想象的脑-机交互算法研究与系统实现技术报告")
    frame = Frame(doc_.leftMargin, doc_.bottomMargin, doc_.width, doc_.height, id="main")
    doc_.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])
    doc_.multiBuild(story)  # 多轮构建以生成目录条目
    print("pdf saved:", out)


if __name__ == "__main__":
    blocks = parse(MD.read_text(encoding="utf-8"))
    build_docx(blocks, DOCX_OUT)
    build_pdf(blocks, PDF_OUT)
