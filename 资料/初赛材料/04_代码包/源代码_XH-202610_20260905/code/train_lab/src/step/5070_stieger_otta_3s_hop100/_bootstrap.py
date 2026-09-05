"""将 07 包与当前包加入 sys.path（07 的 config 由本包 config 覆盖）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
S07 = STEP / "stieger_pseudo_online_3s_hop100"

for p in (str(S07), str(HERE), str(STEP)):
    if p in sys.path:
        sys.path.remove(p)
# HERE 必须优先于 S07，否则 `from config import` 会落到 07 包
sys.path.insert(0, str(STEP))
sys.path.insert(0, str(S07))
sys.path.insert(0, str(HERE))
