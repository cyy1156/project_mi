"""bci2a_2s：EEGNet Task 五折（加权 CE w0=2.2 + Val BalAcc；无 batch balance）。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_eegnet_task import main_cli

if __name__ == "__main__":
    main_cli(
        default_data="bci2a_2s",
        model_name="eegnet_bci2a_2s_wce2p2_balacc",
        window_note="任务 Cue+2~4s；静息下一 Cue 前 2s；500@250Hz",
    )
