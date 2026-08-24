"""Phase4 v2 · 标定/静息试次 → 3s/hop100 窗（无反馈段 · MI 恒满 imagine_s）。

- MI 段 = [t_mi_start, t_mi_end)；标定试次 mi_end = mi_start + 6s
- 游戏试次见 phase4_v2_game.py（可变结束时刻）
- 锚点 t0 ∈ [0.4, dur−3.0]，hop 0.1s；逐窗 z-score（与 openbmi_3s_hop100 同构）

用法：python -m experiment_game.offline.phase4_v2 <session_dir>
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

from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402
from src.common.steps.resample_zscore import trial_zscore  # noqa: E402

FS = 250.0
WIN, HOP, T0_MIN = 3.0, 0.1, 0.4
RAW_COLS = ["C3", "C4", "CZ", "CP3", "CP4", "CPZ", "FC3", "FC4"]
FROZEN = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
REORDER = [RAW_COLS.index(c.upper()) for c in FROZEN]


def load_eeg(session_dir: Path):
    """优先 by_phase 切片；v2 会话无 phase 节点时回退根目录/continuous 全段 eeg。"""
    frames, times = [], []
    by_phase = session_dir / "by_phase"
    if by_phase.is_dir():
        for d in sorted(by_phase.glob("*")):
            f = d / "eeg.csv"
            if not f.exists():
                continue
            with open(f, encoding="utf-8") as fh:
                rd = csv.reader(fh)
                header = next(rd)
                cols = [header.index(c) for c in RAW_COLS]
                rows = [r for r in rd if len(r) == len(header)]
            arr = np.asarray(rows, dtype=np.float64)
            times.append(arr[:, header.index("lsl_time")])
            frames.append(arr[:, cols])
    if not frames:
        for candidate in (session_dir / "eeg.csv", session_dir / "continuous" / "eeg.csv"):
            if not candidate.is_file():
                continue
            with open(candidate, encoding="utf-8") as fh:
                rd = csv.reader(fh)
                header = next(rd)
                cols = [header.index(c) for c in RAW_COLS]
                rows = [r for r in rd if len(r) == len(header)]
            arr = np.asarray(rows, dtype=np.float64)
            if len(arr):
                return arr[:, header.index("lsl_time")], arr[:, cols][:, REORDER]
    if not frames:
        raise RuntimeError("无 eeg.csv（by_phase / 根目录 / continuous 均不可用）")
    return np.concatenate(times), np.concatenate(frames)[:, REORDER]


def cut_segment(x_filt, t_lsl, t_start, t_end):
    """[t_start,t_end) → 锚点 t0∈[0.4, dur−3] 的 z-scored 窗列表 (8,750)。"""
    i0 = int(np.searchsorted(t_lsl, t_start))
    i1 = int(np.searchsorted(t_lsl, t_end))
    dur = (i1 - i0) / FS
    if dur < WIN + T0_MIN - 1e-6:
        return []
    outs = []
    t0 = T0_MIN
    while t0 + WIN <= dur + 1e-9:
        w = trial_zscore(x_filt[i0 + int(round(t0 * FS)) : i0 + int(round((t0 + WIN) * FS))])
        outs.append(w.T.astype(np.float32))
        t0 = round(t0 + HOP, 3)
    return outs


def run(session_dir: str) -> Path:
    sd = Path(session_dir)
    t_lsl, X_raw = load_eeg(sd)
    x = notch_and_bandpass(car_reference(X_raw), FS, l_freq=8.0, h_freq=30.0)
    rows = list(csv.DictReader(open(sd / "alignment" / "trial_table.csv", encoding="utf-8")))

    wins, y_task, y_three, tids = [], [], [], []
    for r in rows:
        if r.get("rejected") == "1" or r.get("invalid") == "1":
            continue
        lab = int(r["label"])  # 0 静息 / 1 左 / 2 右
        phase = r.get("phase", "")
        if phase == "game":
            continue  # 游戏试次走 phase4_v2_game
        segs = []
        if lab == 0:
            # v2 静息试次：label=0 走 mi_start/mi_end（无 rest_start）
            if r.get("t_mi_start") and r.get("t_mi_end"):
                segs.append((float(r["t_mi_start"]), float(r["t_mi_end"]), 0))
            elif r.get("t_rest_start") and r.get("t_rest_end"):
                segs.append((float(r["t_rest_start"]), float(r["t_rest_end"]), 0))
        elif lab in (1, 2) and r.get("t_mi_start") and r.get("t_mi_end"):
            segs.append((float(r["t_mi_start"]), float(r["t_mi_end"]), lab))
            if r.get("t_rest_start"):  # v1：同试次嵌入 Rest
                segs.append((float(r["t_rest_start"]), float(r["t_rest_end"]), 0))
        for t_a, t_b, lb in segs:
            for w in cut_segment(x, t_lsl, t_a, t_b):
                wins.append(w)
                y_task.append(0 if lb == 0 else 1)
                y_three.append(lb)
                tids.append(int(r["trial_id"]))

    out = sd / "phase4_v2"
    out.mkdir(exist_ok=True)
    X = np.stack(wins)[ :, None, :, : ] if wins else np.zeros((0, 1, 8, 750), np.float32)
    np.save(out / "X.npy", X)
    np.save(out / "y_task.npy", np.asarray(y_task, np.int64))
    np.save(out / "y_three.npy", np.asarray(y_three, np.int64))
    np.save(out / "trial_id.npy", np.asarray(tids, np.int64))
    (out / "manifest.json").write_text(json.dumps({
        "win_sec": WIN, "hop_sec": HOP, "t0_min": T0_MIN, "fs": FS,
        "channels": FROZEN, "bandpass_hz": [8.0, 30.0], "zscore": "per-window",
        "n_windows": len(wins), "note": "3s hop100 唯一输出（4s 固定窗已删除 · 2026-08-23）",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{sd.name}: {X.shape} 窗 · 类别 {dict(zip(*np.unique(y_three, return_counts=True)))}")
    return out


if __name__ == "__main__":
    run(sys.argv[1])
