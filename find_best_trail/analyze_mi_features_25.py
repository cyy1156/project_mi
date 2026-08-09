"""find_best_trail 入口：转发到 preprocess_lab 源脚本。

请在仓库根或本目录执行：
  python analyze_mi_features_25.py
或：
  cd code/preprocess_lab
  python -m src.datasets.openbmi.analyze_mi_features_25

默认：OpenBMI 全部 54 被试 · 仅 EEG_MI_train · Rest/MI=cue前/后4s · 2s/hop100 滑窗。
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PRE_ROOT = Path(__file__).resolve().parents[1] / "code" / "preprocess_lab"
if not PRE_ROOT.is_dir():
    raise SystemExit(f"找不到 preprocess_lab：{PRE_ROOT}")
sys.path.insert(0, str(PRE_ROOT))
runpy.run_module("src.datasets.openbmi.analyze_mi_features_25", run_name="__main__")
