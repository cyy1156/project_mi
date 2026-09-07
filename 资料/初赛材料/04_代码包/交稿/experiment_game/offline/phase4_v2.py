"""Phase4 v2 · 标定试次 → OpenBMI-Align 3s/hop100（与 ft_subject_from_v3 / v3 在线同构）。

- Task：Left/Right · [t_cue, t_cue+4s) · Cue 前 0.5s 基线
- Rest：试次间隔（iter_rest_sources_cue_before）
- 跳过 label=0 想象试次（Rest 仅来自间隔段）

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

from experiment_game.core.channel_layout import (  # noqa: E402
    DEVICE_CHANNEL_LABELS,
    reorder_device_to_model_input,
)
from experiment_game.offline.openbmi_align_cut import (  # noqa: E402
    FROZEN,
    FS,
    HOP_SEC,
    WIN_SEC,
    cut_openbmi_align_from_table,
)
from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402

WIN, HOP, T0_MIN = WIN_SEC, HOP_SEC, 0.0  # legacy 导出（replay 工具兼容）
RAW_COLS = list(DEVICE_CHANNEL_LABELS)
# 兼容旧别名：load_eeg 现返回设备序；切窗前再 remap 到 FROZEN
REORDER = [RAW_COLS.index(c.upper()) for c in FROZEN]


def _col_indices(header: list) -> list:
    """按设备序取列；大小写不敏感（兼容历史 Cz/CPz 表头）。"""
    upper = {str(h).upper(): i for i, h in enumerate(header)}
    missing = [c for c in RAW_COLS if c.upper() not in upper]
    if missing:
        raise KeyError(f"eeg.csv 缺少通道 {missing}; cols={header}")
    return [upper[c.upper()] for c in RAW_COLS]


def load_eeg(session_dir: Path):
    """优先 by_phase 切片；回退根目录/continuous。返回 (t, X) 且 X 为设备序 (T,8)。"""
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
                cols = _col_indices(header)
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
                cols = _col_indices(header)
                rows = [r for r in rd if len(r) == len(header)]
            arr = np.asarray(rows, dtype=np.float64)
            if len(arr):
                return arr[:, header.index("lsl_time")], arr[:, cols]
    if not frames:
        raise RuntimeError("无 eeg.csv（by_phase / 根目录 / continuous 均不可用）")
    return np.concatenate(times), np.concatenate(frames)


def cut_segment(x_filt, t_lsl, t_start, t_end):
    """legacy：按 mi 段切窗（旧脚本兼容）；新会话请用 openbmi_align_cut。"""
    from experiment_game.offline.openbmi_align_cut import cue_time_from_row

    row = {"t_cue": t_start, "t_mi_start": t_start}
    t_cue = cue_time_from_row(row)
    if t_cue is None:
        return []
    wins, _, _, _ = cut_openbmi_align_from_table(
        x_filt,
        t_lsl,
        [{"trial_id": 0, "label": 1, "t_cue": t_cue, "t_mi_start": t_cue, "rejected": 0, "invalid": 0}],
        include_rest_interval=False,
    )
    return wins


def run(session_dir: str) -> Path:
    sd = Path(session_dir)
    t_lsl, X_dev = load_eeg(sd)
    # 切窗产物与 FROZEN / OpenBMI 训练轴对齐
    X_raw = reorder_device_to_model_input(X_dev)
    x = notch_and_bandpass(car_reference(X_raw), FS, l_freq=8.0, h_freq=30.0)
    rows = list(csv.DictReader(open(sd / "alignment" / "trial_table.csv", encoding="utf-8")))

    cal_rows = [r for r in rows if str(r.get("phase") or "") == "acquire"]
    wins, y_task, y_three, tids = cut_openbmi_align_from_table(
        x,
        t_lsl,
        cal_rows,
        task_phases={"acquire"},
        include_rest_interval=True,
    )

    out = sd / "phase4_v2"
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
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{sd.name} [cal]: {X.shape} 窗")
    return out


if __name__ == "__main__":
    run(sys.argv[1])
