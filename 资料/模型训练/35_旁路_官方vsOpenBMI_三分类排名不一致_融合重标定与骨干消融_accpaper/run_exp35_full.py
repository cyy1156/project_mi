#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""薄封装：转发到 code/train_lab/.../run_exp35_full.py

用法：
  python run_exp35_full.py
  python run_exp35_full.py --resume
  python run_exp35_full.py --with-h
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[3]
    / "code"
    / "train_lab"
    / "src"
    / "step"
    / "5070_challenge_rankflip_accpaper"
    / "run_exp35_full.py"
)

if not TARGET.is_file():
    # 本机绝对路径兜底
    TARGET = Path(r"D:\MI\code\train_lab\src\step\5070_challenge_rankflip_accpaper\run_exp35_full.py")

if not TARGET.is_file():
    raise SystemExit(f"找不到全量脚本: {TARGET}")

sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
