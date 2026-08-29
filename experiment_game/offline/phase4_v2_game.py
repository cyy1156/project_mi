"""Phase4 v2 · 游戏试次 → OpenBMI-Align 3s/hop100（MI 固定 4s · Cue 锚点）。

- Task：Left/Right · [t_cue, t_cue+4s)
- 与标定 phase4_v2 同切窗逻辑，仅 phase=game

用法：python -m experiment_game.offline.phase4_v2_game <session_dir>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[2] / "code" / "preprocess_lab"))

from experiment_game.offline.openbmi_align_cut import FROZEN, FS, cut_openbmi_align_from_table  # noqa: E402
from experiment_game.experiment.channel_layout import reorder_device_to_model_input  # noqa: E402
from experiment_game.offline.phase4_v2 import load_eeg  # noqa: E402
from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402
from src.common.steps.slide_3s_hop100 import HOP_SEC, WIN_SEC  # noqa: E402

WIN, HOP, T0_MIN = WIN_SEC, HOP_SEC, 0.0


def run(session_dir: str) -> Path:
    sd = Path(session_dir)
    t_lsl, X_dev = load_eeg(sd)
    X_raw = reorder_device_to_model_input(X_dev)
    x = notch_and_bandpass(car_reference(X_raw), FS, l_freq=8.0, h_freq=30.0)
    rows = list(csv.DictReader(open(sd / "alignment" / "trial_table.csv", encoding="utf-8")))
    game_rows = [r for r in rows if r.get("phase") == "game"]

    wins, y_task, y_three, tids = cut_openbmi_align_from_table(
        x,
        t_lsl,
        game_rows,
        include_rest_interval=False,
    )

    out = sd / "phase4_v2_game"
    out.mkdir(exist_ok=True)
    X = np.stack(wins)[:, None, :, :] if wins else np.zeros((0, 1, 8, 750), np.float32)
    np.save(out / "X.npy", X)
    np.save(out / "y_task.npy", np.asarray(y_task, np.int64))
    np.save(out / "y_three.npy", np.asarray(y_three, np.int64))
    np.save(out / "trial_id.npy", np.asarray(tids, np.int64))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "openbmi_align_v1",
                "win_sec": WIN_SEC,
                "hop_sec": HOP_SEC,
                "baseline_before_cue_s": 0.5,
                "task_sec": 4.0,
                "fs": FS,
                "channels": FROZEN,
                "bandpass_hz": [8.0, 30.0],
                "zscore": "per-window",
                "n_windows": len(wins),
                "note": "游戏试次 OpenBMI-Align；与 phase4_v2（标定）并列",
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{sd.name} [game]: {X.shape} 窗")
    return out


if __name__ == "__main__":
    run(sys.argv[1])
