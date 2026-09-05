"""5060 统一包：本包 config 优先，复用 07 的 data_split / infer / metrics。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
S07 = STEP / "stieger_pseudo_online_3s_hop100"

for p in (str(S07), str(HERE), str(STEP)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(STEP))
sys.path.insert(0, str(S07))
sys.path.insert(0, str(HERE))
