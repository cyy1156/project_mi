"""将 Stieger / OTTA / 本包加入 sys.path。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
S07 = STEP / "stieger_pseudo_online_3s_hop100"
OTTA = STEP / "5070_stieger_otta_3s_hop100"
HOP100 = STEP / "baselines_2s_hop100"

for p in (str(STEP), str(S07), str(OTTA), str(HOP100), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
# OTTA 必须最前（otta_infer → OTTA config）；本包用 s25_config 显式导入
for p in (STEP, HOP100, S07, OTTA):
    sys.path.insert(0, str(p))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
