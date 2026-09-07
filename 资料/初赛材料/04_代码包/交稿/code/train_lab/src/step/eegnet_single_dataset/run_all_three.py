"""依次跑四个单库：加权 CE w0=2.2 + Val BalAcc 早停（无 batch balance）。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_eegnet_task import run_experiment

JOBS = [
    ("bci2a_2s", "eegnet_bci2a_2s_wce2p2_balacc", "任务 Cue+2~4s；静息下一 Cue 前 2s；500@250Hz"),
    ("bci2a_4s", "eegnet_bci2a_4s_wce2p2_balacc", "任务 Cue+0~4s；静息下一 Cue 前 4s；1000@250Hz"),
    ("stieger_2s", "eegnet_stieger_2s_wce2p2_balacc", "反馈段最后 2s（Task/Rest 同取法）；500@250Hz"),
    ("stieger_4s", "eegnet_stieger_4s_wce2p2_balacc", "反馈段最后 4s（Task/Rest 同取法）；1000@250Hz"),
]


def main() -> None:
    for data_tag, model_name, note in JOBS:
        print(f"\n########## {model_name} / {data_tag} ##########\n", flush=True)
        run_experiment(data_tag, model_name=model_name, window_note=note)
    print("\nAll four wce2p2+balacc jobs finished.", flush=True)


if __name__ == "__main__":
    main()
