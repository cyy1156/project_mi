#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export submission-facing Markdown (+ Excel) to PDF under 资料/初赛材料.

Usage:
  python export_all_pdfs.py
  python export_all_pdfs.py --skip-report   # skip tech-report (use md2docx_pdf.py)
  python export_all_pdfs.py --skip-excel
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Minimal MD → PDF (reportlab), Chinese fonts via Windows fonts
# ---------------------------------------------------------------------------

IMG_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
HEAD_RE = re.compile(r"^(#{1,3})\s+(.*)$")
TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}")
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)((?:\\.|[^$])+?)(?<!\$)\$(?!\$)")


def _import_math():
    """复用技术报告导出里的 matplotlib mathtext 渲染。"""
    import importlib.util

    p = Path(__file__).resolve().parent / "md2docx_pdf.py"
    spec = importlib.util.spec_from_file_location("md2docx_pdf_math", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    windir = Path(r"C:\Windows\Fonts")
    candidates = [
        ("CN", windir / "msyh.ttc"),
        ("CN", windir / "msyh.ttf"),
        ("CN", windir / "simsun.ttc"),
        ("CNB", windir / "msyhbd.ttc"),
        ("CNB", windir / "simhei.ttf"),
    ]
    registered = {}
    for name, path in candidates:
        if name in registered:
            continue
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(name, str(path), subfontIndex=0))
                registered[name] = path.name
            except Exception:
                continue
    if "CN" not in registered:
        raise RuntimeError("No Chinese TTF found under C:\\Windows\\Fonts")
    if "CNB" not in registered:
        registered["CNB"] = registered["CN"]
        pdfmetrics.registerFont(TTFont("CNB", str(windir / registered["CN"]), subfontIndex=0))
    return "CN", "CNB"


def _latex_to_plain(tex: str) -> str:
    """行内公式 → 中文字体可显示的纯文本（避免 ℝ/𝒟 等缺字小方框）。"""
    s = tex.strip()
    s = re.sub(r"\^{([^}]+)}", r"^(\1)", s)
    s = re.sub(r"_{([^}]+)}", r"_\1", s)
    repl = [
        (r"\mathbb{R}", "R"),
        (r"\mathbb{1}", "1"),
        (r"\mathcal{L}", "L"),
        (r"\mathcal{B}", "B"),
        (r"\mathcal{D}", "D"),
        (r"\mathcal{J}", "J"),
        (r"\theta", "θ"), (r"\tau", "τ"), (r"\pi", "π"), (r"\eta", "η"),
        (r"\mu", "μ"), (r"\sigma", "σ"), (r"\ell", "l"), (r"\beta", "β"),
        (r"\Delta", "Δ"), (r"\nabla", "∇"),
        (r"\times", "x"), (r"\ldots", "..."), (r"\dots", "..."),
        (r"\geq", ">="), (r"\leq", "<="), (r"\ge", ">="), (r"\le", "<="),
        (r"\in", " in "), (r"\mid", "|"), (r"\cdot", "."), (r"\approx", "~"),
        (r"\sum", "sum"), (r"\prod", "prod"), (r"\quad", " "), (r"\qquad", "  "),
        (r"\,", " "), (r"\ ", " "), (r"\log", "log"), (r"\exp", "exp"),
        (r"\max", "max"), (r"\min", "min"),
        (r"\arg\min", "argmin"), (r"\arg\max", "argmax"),
        (r"\mathrm{arg\,min}", "argmin"), (r"\mathrm{arg\,max}", "argmax"),
        (r"\mathrm{src}", "src"), (r"\mathrm{val}", "val"),
        (r"\mathrm{mode}", "mode"), (r"\mathrm{Acc}", "Acc"),
        (r"\tilde", ""), (r"\hat", ""), (r"\frac", "/"),
        (r"\text{--}", "-"), (r"\text", ""), (r"\mathrm", ""),
    ]
    for a, b in repl:
        s = s.replace(a, b)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _esc_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_math_img_tag(tex: str) -> str | None:
    """行内公式渲成 PNG，用 reportlab Paragraph <img> 嵌入（保留上下标）。"""
    mod = _import_math()
    png = mod.render_math(tex, display=False)
    if png is None or not png.exists():
        return None
    try:
        from PIL import Image as PILImage

        with PILImage.open(png) as im:
            # display=False 用 200 dpi；表内/正文约 10–12pt 高
            h = max(9.0, min(13.0, im.height / 200.0 * 72.0 * 0.62))
            w = im.width / max(im.height, 1) * h
        src = str(png.resolve()).replace("\\", "/")
        return f'<img src="{src}" width="{w:.1f}" height="{h:.1f}" valign="middle"/>'
    except Exception as exc:
        print(f"[math-inline] embed failed: {exc}", file=sys.stderr)
        return None


def _format_inline(s: str) -> str:
    """Markdown 行内 → reportlab 富文本；行内公式优先 PNG（角标可见）。"""
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    parts: list[str] = []
    last = 0
    for m in INLINE_MATH_RE.finditer(s):
        parts.append(_esc_xml(s[last:m.start()]))
        tag = _inline_math_img_tag(m.group(1))
        if tag:
            parts.append(tag)
        else:
            parts.append(f"<i>{_esc_xml(_latex_to_plain(m.group(1)))}</i>")
        last = m.end()
    parts.append(_esc_xml(s[last:]))
    out = "".join(parts)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = out.replace("`", "")
    return out.strip()


def _strip_md_inline(s: str) -> str:
    """兼容旧调用：去标记并转行内公式。"""
    return _format_inline(s)


def _append_display_math(story, tex: str, styles, mm, colors) -> None:
    from reportlab.platypus import Image as RLImage, Paragraph, Spacer

    tex = tex.strip()
    if not tex:
        return
    mod = _import_math()
    png = mod.render_math(tex, display=True)
    if png is not None and png.exists():
        try:
            from PIL import Image as PILImage

            with PILImage.open(png) as im:
                max_w = 160 * mm
                w = min(max_w, im.width / 220.0 * 25.4 * mm * 0.92)
                h = im.height / im.width * w
            story.append(Spacer(1, 2 * mm))
            story.append(RLImage(str(png), width=w, height=h))
            story.append(Spacer(1, 2 * mm))
            return
        except Exception as exc:
            print(f"[math] embed failed: {exc}", file=sys.stderr)
    # fallback：可读纯文本，不再甩原始 LaTeX
    story.append(Paragraph(_esc_xml(_latex_to_plain(tex)), styles["BodyCN"]))


def md_to_pdf(md_path: Path, pdf_path: Path, title: str | None = None) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
        Preformatted,
    )
    from reportlab.lib import colors

    cn, cnb = _fonts()
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1CN", fontName=cnb, fontSize=16, leading=22, spaceAfter=10, spaceBefore=14
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2CN", fontName=cnb, fontSize=13, leading=18, spaceAfter=8, spaceBefore=12
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3CN", fontName=cnb, fontSize=11.5, leading=16, spaceAfter=6, spaceBefore=8
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCN", fontName=cn, fontSize=10, leading=15, spaceAfter=4, firstLineIndent=0
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeCN",
            fontName=cn,
            fontSize=8.5,
            leading=11,
            backColor=colors.Color(0.95, 0.95, 0.95),
        )
    )
    styles.add(ParagraphStyle(name="CellCN", fontName=cn, fontSize=8, leading=11))

    story = []
    if title:
        story.append(Paragraph(_format_inline(title), styles["H1CN"]))
        story.append(Spacer(1, 4 * mm))

    i = 0
    in_code = False
    code_buf: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                block = "\n".join(code_buf) if code_buf else " "
                story.append(Preformatted(block[:8000], styles["CodeCN"]))
                story.append(Spacer(1, 2 * mm))
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # 独立 $$ … $$ 公式块 → PNG
        if line.strip() == "$$":
            buf: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                buf.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == "$$":
                i += 1
            _append_display_math(story, "\n".join(buf), styles, mm, colors)
            continue
        if line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 4:
            _append_display_math(story, line.strip()[2:-2], styles, mm, colors)
            i += 1
            continue

        m = HEAD_RE.match(line)
        if m:
            level = len(m.group(1))
            body = _format_inline(m.group(2))
            key = {1: "H1CN", 2: "H2CN", 3: "H3CN"}.get(level, "H3CN")
            story.append(Paragraph(body, styles[key]))
            i += 1
            continue

        if line.strip().startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.search(lines[i + 1] or ""):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip().strip("|")
                if TABLE_SEP_RE.search(lines[i]):
                    i += 1
                    continue
                cells = [_format_inline(c) for c in raw.split("|")]
                rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                norm = [r + [""] * (ncol - len(r)) for r in rows]
                data = [[Paragraph(c, styles["CellCN"]) for c in r] for r in norm]
                tw = 170 * mm
                col_w = [tw / ncol] * ncol
                t = Table(data, colWidths=col_w, repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.95)),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 3 * mm))
            continue

        if not line.strip():
            story.append(Spacer(1, 1.5 * mm))
            i += 1
            continue

        if IMG_RE.match(line.strip()):
            story.append(Paragraph(_format_inline(line), styles["BodyCN"]))
            i += 1
            continue

        if line.strip().startswith(">"):
            story.append(Paragraph(_format_inline(line.lstrip("> ").strip()), styles["BodyCN"]))
            i += 1
            continue

        story.append(Paragraph(_format_inline(line), styles["BodyCN"]))
        i += 1

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title or md_path.stem,
    )
    doc.build(story)
    print(f"[pdf] {pdf_path} ({pdf_path.stat().st_size} bytes)")


def excel_to_pdf(xlsx: Path, pdf_path: Path) -> bool:
    """Prefer LibreOffice; fallback: dump first sheets via openpyxl+reportlab."""
    import shutil
    import subprocess

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        out_dir = pdf_path.parent
        cmd = [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(xlsx),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
            produced = out_dir / (xlsx.stem + ".pdf")
            if produced.exists() and produced != pdf_path:
                produced.replace(pdf_path)
            if pdf_path.exists():
                print(f"[pdf-excel] {pdf_path} via LibreOffice")
                return True
        except Exception as exc:
            print(f"[pdf-excel] LibreOffice failed: {exc}")

    try:
        from openpyxl import load_workbook
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors

        cn, cnb = _fonts()
        wb = load_workbook(xlsx, data_only=True, read_only=True)
        styles_cell = ParagraphStyle(name="XC", fontName=cn, fontSize=7, leading=9)
        styles_h = ParagraphStyle(name="XH", fontName=cnb, fontSize=11, leading=14, spaceAfter=6)

        story = []
        for si, name in enumerate(wb.sheetnames[:10]):
            ws = wb[name]
            story.append(Paragraph(f"工作表：{name}", styles_h))
            rows = []
            for ri, row in enumerate(ws.iter_rows(values_only=True)):
                if ri > 80:
                    break
                cells = []
                for v in list(row)[:12]:
                    if v is None:
                        cells.append("")
                    else:
                        s = str(v)
                        cells.append(s[:80])
                if any(c.strip() for c in cells):
                    rows.append([Paragraph(c, styles_cell) for c in cells])
            if rows:
                ncol = max(len(r) for r in rows)
                for r in rows:
                    while len(r) < ncol:
                        r.append(Paragraph("", styles_cell))
                tw = 260 * mm
                t = Table(rows, colWidths=[tw / ncol] * ncol, repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.95)),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )
                story.append(t)
            story.append(Spacer(1, 6 * mm))
        wb.close()
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=landscape(A4),
            leftMargin=10 * mm,
            rightMargin=10 * mm,
            topMargin=12 * mm,
            bottomMargin=12 * mm,
            title=xlsx.stem,
        )
        doc.build(story)
        print(f"[pdf-excel] {pdf_path} via openpyxl fallback")
        return True
    except Exception as exc:
        print(f"[pdf-excel] FAILED {xlsx}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--skip-excel", action="store_true")
    args = ap.parse_args()

    jobs = [
        (ROOT / "03_演示视频" / "演示视频脚本_分镜与解说.md",
         ROOT / "03_演示视频" / "演示视频脚本_分镜与解说.pdf",
         "在线系统演示视频 · 分镜与解说词（XH-202610）"),
        (ROOT / "04_代码包" / "README_代码包说明.md",
         ROOT / "04_代码包" / "交稿" / "README_代码包说明.pdf",
         "源代码包说明（XH-202610）"),
        (ROOT / "04_代码包" / "00_目录结构与复现总览.md",
         ROOT / "04_代码包" / "交稿" / "00_目录结构与复现总览.pdf",
         "00 · 目录结构与复现总览"),
        (ROOT / "04_代码包" / "01_数据集获取说明.md",
         ROOT / "04_代码包" / "交稿" / "01_数据集获取说明.pdf",
         "01 · 数据集获取说明"),
        (ROOT / "04_代码包" / "附录A_实验证据链索引.md",
         ROOT / "04_代码包" / "交稿" / "02_附录A_实验证据链索引.pdf",
         "附录A · 实验证据链索引"),
        (ROOT / "04_代码包" / "附录B_自采数据质控样例.md",
         ROOT / "04_代码包" / "交稿" / "03_附录B_自采数据质控样例.pdf",
         "附录B · 自采数据质控样例"),
        (ROOT / "04_代码包" / "04_附录C_算法公式与算法详细介绍.md",
         ROOT / "04_代码包" / "交稿" / "04_附录C_算法公式与算法详细介绍.pdf",
         "附录C · 算法公式与算法详细介绍"),
        (ROOT / "04_代码包" / "README_采集软件说明.md",
         ROOT / "04_代码包" / "交稿" / "README_采集软件说明.pdf",
         "采集软件说明"),
        # 02 离线验证以 Excel 提交（结果+数据集说明），不导出说明 PDF
        (ROOT / "README_初赛材料总览.md",
         ROOT / "_build" / "out" / "README_初赛材料总览.pdf",
         "初赛材料总览（内部）"),
    ]

    # Also copy appendix A md into 交稿 numbered name if missing
    app_a_src = ROOT / "04_代码包" / "附录A_实验证据链索引.md"
    app_a_dst = ROOT / "04_代码包" / "交稿" / "02_附录A_实验证据链索引.md"
    if app_a_src.exists():
        app_a_dst.write_text(app_a_src.read_text(encoding="utf-8"), encoding="utf-8")

    for md, pdf, title in jobs:
        if not md.exists():
            print(f"[skip] missing {md}")
            continue
        try:
            md_to_pdf(md, pdf, title=title)
        except Exception as exc:
            print(f"[FAIL] {md}: {exc}")

    if not args.skip_excel:
        # 离线验证官方件为 Excel；不再导出 PDF 充当提交件
        xlsx = ROOT / "02_离线验证" / "交稿" / "离线性能验证报告_XH-202610.xlsx"
        if not xlsx.exists():
            xlsx = ROOT / "02_离线验证" / "离线性能验证报告_XH-202610.xlsx"
        if xlsx.exists():
            print(f"[skip-pdf] 离线验证以 Excel 提交：{xlsx.name}（不导出 PDF）")
        else:
            print("[skip] excel missing")

    if not args.skip_report:
        report_script = Path(__file__).resolve().parent / "md2docx_pdf.py"
        if report_script.exists():
            import runpy

            print("[report] running md2docx_pdf.py …")
            sys.argv = [str(report_script)]
            runpy.run_path(str(report_script), run_name="__main__")
        else:
            print("[report] md2docx_pdf.py not found")

    print("[done] export_all_pdfs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
