# -*- coding: utf-8 -*-
"""Audit submission package: 上海理工大学_面向少样本个性化的智能运动想象脑机交互系统."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from pypdf import PdfReader

ROOT = Path(r"D:/MI/资料/初赛材料/上海理工大学_面向少样本个性化的智能运动想象脑机交互系统")
SRC = Path(r"D:/MI/资料/初赛材料")  # working tree for comparison

issues: list[tuple[str, str]] = []  # (level, msg)
notes: list[str] = []


def ok(msg: str):
    notes.append(f"[OK] {msg}")


def warn(msg: str):
    issues.append(("WARN", msg))


def fail(msg: str):
    issues.append(("FAIL", msg))


def info(msg: str):
    notes.append(f"[INFO] {msg}")


def sha256(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        if limit:
            h.update(f.read(limit))
        else:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
    return h.hexdigest()[:16]


def docx_text(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    return "".join(
        (el.text or "")
        for el in root.iter()
        if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == "t"
    )


def pdf_text(p: Path, max_pages: int | None = None) -> str:
    r = PdfReader(str(p))
    pages = r.pages if max_pages is None else r.pages[:max_pages]
    return "\n".join((pg.extract_text() or "") for pg in pages)


def main():
    print("=" * 72)
    print("SUBMIT PACKAGE AUDIT")
    print(ROOT)
    print("=" * 72)

    # ---------- 1. required top-level ----------
    expected_dirs = {
        "01技术报告": None,
        "02_离线验证": None,
        "03_演示视频": None,
        "04_补充材料_代码包": None,
        "05_报名表": None,
    }
    # resolve actual names (encoding-safe)
    children = {p.name: p for p in ROOT.iterdir()}
    print("\n## 1. Top-level structure")
    for name in children:
        print(f"  - {name}/" if children[name].is_dir() else f"  - {name}")

    d01 = next(p for n, p in children.items() if n.startswith("01"))
    d02 = next(p for n, p in children.items() if "离线" in n or n.startswith("02"))
    d03 = next(p for n, p in children.items() if "演示" in n or n.startswith("03"))
    d04 = next(p for n, p in children.items() if "代码" in n or n.startswith("04"))
    d05 = next(p for n, p in children.items() if "报名" in n or n.startswith("05"))
    ok(f"five sections present: {d01.name}, {d02.name}, {d03.name}, {d04.name}, {d05.name}")

    # ---------- 2. tech report ----------
    print("\n## 2. Tech report (01)")
    md = next(d01.glob("*.md"))
    docx = next(d01.glob("*.docx"))
    pdf = next(d01.glob("*.pdf"))
    info(f"MD   {md.name} {md.stat().st_size} mtime")
    info(f"DOCX {docx.name} {docx.stat().st_size}")
    info(f"PDF  {pdf.name} {pdf.stat().st_size} pages={len(PdfReader(str(pdf)).pages)}")

    md_txt = md.read_text(encoding="utf-8")
    dx_txt = docx_text(docx)
    pf_txt = pdf_text(pdf)

    anchors = [
        "申报信息",
        "上海理工大学",
        "中国智慧工程研究会",
        "面向少样本个性化的智能运动想象脑机交互系统",
        "陈越云",
        "尹钟",
        "XH-202610",
        "61.88%",
        "51.11%",
        "47.60%",
        "92.40%",
        "CausalFuse-8",
        "QuadFold-59",
        "T-Shallow",
        "Shallow-b",
        "第二浅层差异",
        "t0 加权",
    ]
    for a in anchors:
        flags = (a in md_txt, a.replace(" ", "") in re.sub(r"\s+", "", dx_txt) or a in dx_txt, a.replace(" ", "") in re.sub(r"\s+", "", pf_txt) or a in pf_txt)
        if all(flags):
            ok(f"report anchor '{a}' in MD/DOCX/PDF")
        else:
            fail(f"report anchor '{a}' missing: MD={flags[0]} DOCX={flags[1]} PDF={flags[2]}")

    # compare to working 交稿
    src_md = SRC / "01_技术报告" / "交稿" / "技术报告_XH-202610_v4.md"
    src_docx = SRC / "01_技术报告" / "交稿" / "技术报告_XH-202610.docx"
    src_pdf = SRC / "01_技术报告" / "交稿" / "技术报告_XH-202610.pdf"
    if src_md.exists():
        if sha256(md) == sha256(src_md):
            ok("01 MD == 交稿 v4.md (sha256)")
        else:
            warn(f"01 MD differs from 交稿 v4.md (pack={sha256(md)} src={sha256(src_md)})")
    if src_docx.exists():
        if sha256(docx) == sha256(src_docx):
            ok("01 DOCX == 交稿 docx")
        else:
            warn(f"01 DOCX differs from 交稿 (pack={sha256(docx)} src={sha256(src_docx)} size {docx.stat().st_size}/{src_docx.stat().st_size})")
    if src_pdf.exists():
        if sha256(pdf) == sha256(src_pdf):
            ok("01 PDF == 交稿 pdf")
        else:
            warn(f"01 PDF differs from 交稿 (pack={sha256(pdf)} src={sha256(src_pdf)} size {pdf.stat().st_size}/{src_pdf.stat().st_size})")

    # placeholder leftovers
    for bad in ["【图位预留】", "TODO", "TBD", "xxx", "FIXME"]:
        if bad in md_txt:
            warn(f"MD still contains placeholder '{bad}'")

    # ---------- 3. offline ----------
    print("\n## 3. Offline validation (02)")
    xlsx = list(d02.glob("*.xlsx"))
    csvs = list(d02.glob("*.csv")) + list((d02 / "原始验证数据").glob("*.csv") if (d02 / "原始验证数据").exists() else [])
    # find folders
    raw_dir = next((p for p in d02.iterdir() if p.is_dir() and ("原始" in p.name or "验证数据" in p.name)), None)
    shot_dir = next((p for p in d02.iterdir() if p.is_dir() and ("截图" in p.name or "过程" in p.name)), None)

    if not xlsx:
        fail("missing 离线性能验证报告 xlsx")
    else:
        ok(f"xlsx present: {xlsx[0].name} ({xlsx[0].stat().st_size} bytes)")
        src_xlsx = SRC / "02_离线验证" / "交稿" / "离线性能验证报告_XH-202610.xlsx"
        if not src_xlsx.exists():
            src_xlsx = SRC / "02_离线验证" / "离线性能验证报告_XH-202610.xlsx"
        if src_xlsx.exists():
            if sha256(xlsx[0]) == sha256(src_xlsx):
                ok("02 xlsx == working copy")
            else:
                warn(f"02 xlsx hash differs from working ({sha256(xlsx[0])} vs {sha256(src_xlsx)})")

    sub_csv = d02 / "sample_submission.csv"
    if not sub_csv.exists() and raw_dir:
        alt = list(raw_dir.glob("sample_submission.csv"))
        sub_csv = alt[0] if alt else sub_csv
    if sub_csv.exists():
        with sub_csv.open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        ok(f"sample_submission.csv rows={len(rows)} cols={list(rows[0].keys()) if rows else []}")
        if len(rows) != 120:
            fail(f"sample_submission expected 120 rows, got {len(rows)}")
        # label distribution
        from collections import Counter

        # guess label col
        label_key = None
        for k in rows[0].keys():
            if k.lower() in ("label", "y", "prediction", "pred", "class"):
                label_key = k
                break
        if label_key is None:
            # second column often label
            label_key = list(rows[0].keys())[-1]
        counts = Counter(r[label_key] for r in rows)
        info(f"label col='{label_key}' counts={dict(counts)}")
        if set(counts) - {"0", "1", "2", 0, 1, 2}:
            warn(f"unexpected labels: {counts}")
    else:
        fail("sample_submission.csv missing")

    if shot_dir:
        shots = list(shot_dir.glob("*.png")) + list(shot_dir.glob("*.jpg"))
        ok(f"verification screenshots: {len(shots)} in {shot_dir.name}")
        if len(shots) < 7:
            warn(f"expected ~7 screenshots (S01-S07), found {len(shots)}")
    else:
        warn("no 验证过程截图 folder")

    if raw_dir:
        raw_files = list(raw_dir.rglob("*"))
        info(f"原始验证数据 files: {sum(1 for p in raw_files if p.is_file())}")
    else:
        warn("no 原始验证数据 folder")

    # ---------- 4. video ----------
    print("\n## 4. Demo video (03)")
    vids = list(d03.glob("*.mp4")) + list(d03.glob("*.MP4")) + list(d03.glob("*.mkv"))
    if not vids:
        fail("demo video missing")
    else:
        v = vids[0]
        mb = v.stat().st_size / (1024 * 1024)
        ok(f"video {v.name} size={mb:.1f} MB")
        if mb < 1:
            fail("video file suspiciously small")
        # duration via mutagen or ffprobe if available
        dur = None
        try:
            import subprocess

            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode == 0 and r.stdout.strip():
                dur = float(r.stdout.strip())
        except Exception:
            pass
        if dur is None:
            try:
                from mutagen.mp4 import MP4

                dur = float(MP4(str(v)).info.length)
            except Exception:
                pass
        if dur is not None:
            info(f"video duration={dur:.1f}s ({dur/60:.2f} min)")
            if dur > 10 * 60 + 5:
                fail(f"video exceeds 10 min limit: {dur/60:.2f} min")
            else:
                ok("video duration ≤ 10 min")
        else:
            warn("could not read video duration (install ffprobe/mutagen)")

    # ---------- 5. code pack ----------
    print("\n## 5. Code pack (04)")
    req = d04 / "requirements.txt"
    if req.exists():
        ok(f"requirements.txt ({req.stat().st_size} B)")
    else:
        warn("requirements.txt missing at code-pack root")

    pdfs04 = list(d04.glob("*.pdf"))
    info(f"code-pack root PDFs: {len(pdfs04)}")
    for p in sorted(pdfs04, key=lambda x: x.name):
        info(f"  PDF {p.name} ({p.stat().st_size})")

    for must_dir in ["code", "experiment_game", "collect_data"]:
        p = d04 / must_dir
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            ok(f"{must_dir}/ present ({n} files)")
        else:
            fail(f"{must_dir}/ missing")

    # secrets / credentials scan
    secret_hits = []
    pat = re.compile(
        r"(api[_-]?key|secret|password|passwd|private[_-]?key|BEGIN RSA|sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})",
        re.I,
    )
    token_files = []
    for p in d04.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".pt", ".pth", ".pkl", ".npy", ".npz", ".pdf", ".xlsx"}:
            continue
        if p.stat().st_size > 2_000_000:
            continue
        name_l = p.name.lower()
        if any(x in name_l for x in ["token", "secret", "credential", ".env", "password"]):
            token_files.append(p)
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pat.search(txt) and p.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".txt", ".md", ".env", ".ini"}:
            # filter common false positives
            if "password" in txt.lower() and "password=" not in txt.lower() and "passwd" not in txt.lower():
                if not re.search(r"password\s*[:=]\s*['\"][^'\"]+['\"]", txt, re.I):
                    continue
            secret_hits.append(str(p.relative_to(d04)))

    if token_files:
        for t in token_files:
            rel = t.relative_to(d04)
            preview = t.read_text(encoding="utf-8", errors="ignore")[:80].replace("\n", " ")
            warn(f"possible secret/token file: {rel} | preview={preview!r}")
    if secret_hits:
        for h in secret_hits[:20]:
            warn(f"possible secret pattern in {h}")
    else:
        ok("no obvious hardcoded API key / AWS key patterns in text configs")

    # junk / patch scripts
    junk = []
    for p in d04.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if n.startswith("_rewrite") or n.startswith("_patch") or n.startswith("_fix") or n.startswith("_bootstrap"):
            junk.append(str(p.relative_to(d04)))
        if n.endswith(".pyc") or "__pycache__" in str(p):
            junk.append(str(p.relative_to(d04)))
    if junk:
        warn(f"dev/junk files in pack ({len(junk)}), e.g. {junk[:8]}")
    else:
        ok("no obvious _patch/_fix/__pycache__ junk")

    # experiment_game slim check
    eg = d04 / "experiment_game"
    if eg.exists():
        machines = eg / "machines"
        if machines.exists():
            warn("experiment_game/machines still present (should be slimmed)")
        else:
            ok("experiment_game/machines absent (slimmed)")
        cfg = eg / "config" / "e1f_four_member.json"
        if cfg.exists():
            conf = json.loads(cfg.read_text(encoding="utf-8"))
            info(f"e1f_four_member.json keys={list(conf)[:12]}")
            ok("e1f_four_member.json present")
        else:
            fail("e1f_four_member.json missing")

    # ---------- 6. registration form ----------
    print("\n## 6. Registration form (05)")
    forms = list(d05.glob("*.pdf")) + list(d05.glob("*.PDF"))
    if not forms:
        fail("报名表 PDF missing")
    else:
        f = forms[0]
        ok(f"报名表 {f.name} ({f.stat().st_size/1024:.0f} KB)")
        if f.stat().st_size < 50_000:
            warn("报名表 PDF very small; check scan quality")

    # ---------- 7. cross-number consistency ----------
    print("\n## 7. Cross-material number check")
    corpus = md_txt + "\n" + dx_txt + "\n" + pf_txt
    # also try readme pdfs text (first pages)
    for p in pdfs04:
        try:
            corpus += "\n" + pdf_text(p, max_pages=3)
        except Exception:
            pass

    must_nums = {
        "61.88": "OpenBMI",
        "51.11": "指定集留一",
        "47.60": "真人均值",
        "92.40": "真人最优",
        "66.30": "BCI2a",
        "65.90": "Stieger微调",
        "1.11": "前向ms",
    }
    for num, label in must_nums.items():
        if num in md_txt:
            ok(f"number {num} ({label}) in tech-report MD")
        else:
            fail(f"number {num} ({label}) missing in tech-report MD")

    # ---------- summary ----------
    print("\n" + "=" * 72)
    print("NOTES")
    for n in notes:
        print(n)
    print("\nISSUES")
    if not issues:
        print("[NONE]")
    else:
        for lv, msg in issues:
            print(f"[{lv}] {msg}")
    fails = sum(1 for lv, _ in issues if lv == "FAIL")
    warns = sum(1 for lv, _ in issues if lv == "WARN")
    print("\n" + "=" * 72)
    print(f"RESULT: {fails} FAIL, {warns} WARN, {len(notes)} OK/INFO")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    import re
    import sys

    sys.exit(main())
