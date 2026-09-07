"""把 baselines_2s_hop100 挂到 sys.path（append），供 feat_bandpower / load_external / raw_time。

必须 append 而非 insert，避免覆盖本包 task_runner。
"""

from __future__ import annotations

import sys
from pathlib import Path

HOP100 = Path(__file__).resolve().parent.parent / "baselines_2s_hop100"
_sp = str(HOP100)
if _sp not in sys.path:
    sys.path.append(_sp)
