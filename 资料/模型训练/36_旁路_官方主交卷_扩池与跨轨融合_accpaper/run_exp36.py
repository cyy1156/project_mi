#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""薄封装 → code/train_lab/.../run_exp36.py"""
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
    / "5070_challenge_exp36_pool_xtrack_accpaper"
    / "run_exp36.py"
)
if not TARGET.is_file():
    TARGET = Path(r"D:\MI\code\train_lab\src\step\5070_challenge_exp36_pool_xtrack_accpaper\run_exp36.py")
sys.argv = [str(TARGET), *sys.argv[1:]]
runpy.run_path(str(TARGET), run_name="__main__")
