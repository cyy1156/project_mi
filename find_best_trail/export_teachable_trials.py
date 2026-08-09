"""转发 → preprocess_lab.export_teachable_trials（方案 06 · B0）。"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "code" / "preprocess_lab"
sys.path.insert(0, str(PRE))
runpy.run_module("src.datasets.openbmi.export_teachable_trials", run_name="__main__")
