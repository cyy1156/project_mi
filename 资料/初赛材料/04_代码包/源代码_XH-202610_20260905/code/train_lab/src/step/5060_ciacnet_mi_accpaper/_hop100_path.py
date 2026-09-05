"""把正式 hop100 Acc_paper 包挂到 path（append），供 runner 依赖。"""
from __future__ import annotations

import sys
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parent.parent / "5060_baselines_openbmi_2s_hop100_accpaper"
HOP100 = Path(__file__).resolve().parent.parent / "baselines_2s_hop100"
for p in (HOP100, OFFICIAL):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)
