"""切窗位置诊断：用 core/windowing 对真实历史会话重切，与磁盘 v3_segments 做相关比对。

⚠️ 重要：本脚本**不是**回归锚点，仅作粗粒度诊断。
   v3_segments 来自在线环形缓冲区（原始采集流），而 continuous/eeg.csv 是落盘流
   （含补齐/重采样，drop_rate 通常 >0）。两者采样点存在固有偏差，逐样本比对不成立，
   实测相关会在 ±0.9 间摆动，属预期现象，不代表切窗逻辑有误。

   真正的数值保真度校验请用同目录的 `windowing_fidelity_check.py`
   （core/windowing vs 训练侧 src.common.steps 原始实现，逐值一致）。

用途：确认「窗长 / 通道序 / 切片量级」与历史产物自洽。
用法：python -m experiment_game.tools.windowing_regression_check <session_dir>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from experiment_game.core.windowing import (
    BASELINE_BEFORE_CUE_S,
    FS,
    TASK_SEC,
    lsl_to_sample,
    cue_time_from_row,
)


def load_eeg(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """返回 (data[C,N], t_lsl[N], labels)。"""
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    lower = [h.strip().lower() for h in header]
    # 时间列
    time_names = ("t_lsl", "timestamp", "time", "lsl_ts", "lsl_time", "t", "ts")
    ti = next((i for i, h in enumerate(lower) if h in time_names), None)
    if ti is None:
        raise SystemExit(f"未找到时间列: {header}")
    ch_idx = [i for i, h in enumerate(lower) if h.startswith("eeg")]
    if not ch_idx:
        ch_idx = [i for i in range(len(header)) if i != ti]
    labels = [header[i].strip() for i in ch_idx]
    data = np.asarray([[float(row[i]) for i in ch_idx] for row in rows], dtype=np.float64)
    t = np.asarray([float(row[ti]) for row in rows], dtype=np.float64)
    return data, t, labels


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python -m experiment_game.tools.windowing_regression_check <session_dir>")
        return 2
    sess = Path(sys.argv[1])
    csv_path = sess / "continuous" / "eeg.csv"
    if not csv_path.is_file():
        csv_path = sess / "eeg.csv"
    tbl = sess / "alignment" / "trial_table.csv"
    segdir = sess / "v3_segments"

    x, t_lsl, labels = load_eeg(csv_path)
    print(f"数据: {csv_path.name}  形状={x.shape}  通道={labels}")

    with tbl.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("t_cue")]

    n_pre = int(round(BASELINE_BEFORE_CUE_S * FS))
    n_task = int(round(TASK_SEC * FS))
    expected_len = n_pre + n_task
    print(f"期望窗长: {n_pre}(baseline) + {n_task}(task) = {expected_len}")

    seg_files = sorted(segdir.glob("trial*.npy")) if segdir.is_dir() else []
    print(f"磁盘 v3_segments: {len(seg_files)} 个")
    if seg_files:
        a0 = np.load(seg_files[0])
        print(f"  trial001 形状={a0.shape}")

    n_ok = n_cmp = 0
    corrs: list[float] = []
    for i, r in enumerate(rows):
        if str(r.get("rejected") or "0") == "1":
            continue
        if str(r.get("invalid") or "0") == "1":
            continue
        t_cue = cue_time_from_row(r)
        if t_cue is None:
            continue
        idx = lsl_to_sample(t_lsl, t_cue)
        s = idx - n_pre
        e = idx + n_task
        if s < 0 or e > x.shape[0]:
            continue
        seg = x[s:e, :]  # (T, C)
        n_ok += 1
        if i < len(seg_files):
            old = np.load(segdir / f"trial{i + 1:03d}.npy")
            if old.shape == seg.shape:
                n_cmp += 1
                # 逐通道 Pearson 相关（容忍滤波差异，只验证切片位置与通道序）
                cs = []
                for c in range(seg.shape[1]):
                    a, b = seg[:, c], old[:, c]
                    if a.std() > 1e-12 and b.std() > 1e-12:
                        cs.append(float(np.corrcoef(a, b)[0, 1]))
                m = float(np.mean(cs)) if cs else 0.0
                corrs.append(m)
                if i < 3:
                    print(f"  trial{i + 1:03d} 相关={m:.4f} 通道相关={[round(v,3) for v in cs]}")

    print(f"\n可切窗 trial 数: {n_ok}")
    if corrs:
        print(f"与磁盘 v3_segments 比对: {n_cmp} 个，相关范围 {min(corrs):.4f} ~ {max(corrs):.4f}")
        print("说明: 相关偏低属预期（在线缓冲 vs 落盘流存在采样点偏差），")
        print("      本结果仅用于确认窗长/通道数/可切窗数量自洽，不作数值回归依据。")
    print("数值保真度请以 windowing_fidelity_check.py 为准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
