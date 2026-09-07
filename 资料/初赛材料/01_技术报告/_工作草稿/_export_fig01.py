# -*- coding: utf-8 -*-
"""Export aligned architecture SVG from HTML draft to fig01 PNG via Edge headless."""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "在线系统实时交互架构框架图.html"
EXPORT_HTML = ROOT / "_fig01_export.html"
SHOT_DIR = ROOT / "_fig01_shot"
EDGE = Path(r"D:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

FIG_DIRS = [
    ROOT.parent / "figures",
    ROOT.parent / "交稿" / "figures",
]


def main() -> None:
    src = SRC.read_text(encoding="utf-8")
    m = re.search(r'(<svg viewBox="0 0 1200 1140".*?</svg>)', src, re.S)
    if not m:
        raise SystemExit("svg block not found")
    svg = m.group(1)
    EXPORT_HTML.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"UTF-8\">\n"
        "<style>\n"
        "html,body{margin:0;padding:0;background:#fff;}\n"
        ".card{padding:12px 16px 8px;background:#fff;}\n"
        "svg{display:block;width:1200px;height:1140px;}\n"
        'svg text{font-family:"Microsoft YaHei","PingFang SC",sans-serif;}\n'
        "</style></head><body><div class=\"card\">"
        + svg
        + "</div></body></html>\n",
        encoding="utf-8",
    )
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    shot_png = SHOT_DIR / "shot2x.png"
    if shot_png.exists():
        shot_png.unlink()

    url = EXPORT_HTML.resolve().as_uri()
    cmd = [
        str(EDGE),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--window-size=1240,1200",
        f"--screenshot={shot_png}",
        url,
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=90)
    for _ in range(30):
        if shot_png.exists() and shot_png.stat().st_size > 20_000:
            break
        time.sleep(0.2)
    if not shot_png.exists():
        raise SystemExit(f"screenshot missing: {shot_png}")

    im = Image.open(shot_png).convert("RGB")
    arr = np.asarray(im)
    mask = (arr < 250).any(axis=2)
    ys, xs = np.where(mask)
    crop = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    print("crop size:", crop.size)

    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        dst = d / "fig01_system_architecture.png"
        crop.save(dst, "PNG", optimize=True)
        print("copied ->", dst, dst.stat().st_size)

    # inspection crops
    w, h = crop.size
    crop.crop((int(w * 0.04), int(h * 0.36), int(w * 0.36), int(h * 0.50))).save(
        ROOT / "_check_box6.png"
    )
    crop.crop((int(w * 0.34), int(h * 0.48), int(w * 0.68), int(h * 0.60))).save(
        ROOT / "_check_box10.png"
    )
    crop.crop((int(w * 0.02), int(h * 0.76), int(w * 0.99), int(h * 0.99))).save(
        ROOT / "_check_bottom.png"
    )
    print("checks written")


if __name__ == "__main__":
    main()
