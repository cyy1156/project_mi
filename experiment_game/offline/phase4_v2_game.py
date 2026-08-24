"""Phase4 v2 · 游戏试次 → 3s/hop100 窗（有反馈段 · 可变 MI 结束时刻）。

- MI 段 = [t_mi_start, t_mi_end)；t_mi_end = touch/reach 时刻或满 6s（由 trial_v2 事件落盘）
- 短于 3s+0.4s 的试次跳过（无法切出合规窗）
- 输出目录：phase4_v2_game/（与 phase4_v2/ 并列，后处理可 merge）

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

from experiment_game.offline.phase4_v2 import (  # noqa: E402
    FS, FROZEN, WIN, HOP, T0_MIN,
    cut_segment, load_eeg,
)
from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402


def run(session_dir: str) -> Path:
    sd = Path(session_dir)
    t_lsl, X_raw = load_eeg(sd)
    x = notch_and_bandpass(car_reference(X_raw), FS, l_freq=8.0, h_freq=30.0)
    rows = list(csv.DictReader(open(sd / "alignment" / "trial_table.csv", encoding="utf-8")))

    wins, y_task, y_three, tids = [], [], [], []
    for r in rows:
        if r.get("rejected") == "1" or r.get("invalid") == "1":
            continue
        if r.get("phase") != "game":
            continue
        lab = int(r["label"])
        if lab not in (1, 2) or not r.get("t_mi_start") or not r.get("t_mi_end"):
            continue
        t_a, t_b = float(r["t_mi_start"]), float(r["t_mi_end"])
        for w in cut_segment(x, t_lsl, t_a, t_b):
            wins.append(w)
            y_task.append(1)
            y_three.append(lab)
            tids.append(int(r["trial_id"]))

    out = sd / "phase4_v2_game"
    out.mkdir(exist_ok=True)
    X = np.stack(wins)[:, None, :, :] if wins else np.zeros((0, 1, 8, 750), np.float32)
    np.save(out / "X.npy", X)
    np.save(out / "y_task.npy", np.asarray(y_task, np.int64))
    np.save(out / "y_three.npy", np.asarray(y_three, np.int64))
    np.save(out / "trial_id.npy", np.asarray(tids, np.int64))
    (out / "manifest.json").write_text(json.dumps({
        "win_sec": WIN, "hop_sec": HOP, "t0_min": T0_MIN, "fs": FS,
        "channels": FROZEN, "bandpass_hz": [8.0, 30.0], "zscore": "per-window",
        "n_windows": len(wins),
        "note": "游戏试次可变 MI 结束；与 phase4_v2（标定满 6s）并列",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{sd.name} [game]: {X.shape} 窗")
    return out


if __name__ == "__main__":
    run(sys.argv[1])
