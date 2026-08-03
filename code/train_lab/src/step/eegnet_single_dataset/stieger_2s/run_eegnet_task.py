"""stieger_2s：EEGNet Task 五折（加权 CE w0=2.2 + Val BalAcc；无 batch balance）。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_eegnet_task import main_cli

if __name__ == "__main__":
    main_cli(
        default_data="stieger_2s",
        model_name="eegnet_stieger_2s_wce2p2_balacc",
        window_note="反馈段最后 2s（Task/Rest 同取法）；500@250Hz",
    )
