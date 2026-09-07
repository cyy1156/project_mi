"""OpenBMI-Align 切窗（薄包装）：表策略在此，数学在 core.windowing。

权威实现：experiment_game.core.windowing
本模块保留 trial_table / rest 打点策略与向后兼容导出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiment_game.core.windowing import (
    BASELINE_BEFORE_CUE_S,
    FROZEN,
    FS,
    HOP_SEC,
    N_TIMES,
    TASK_SEC,
    WIN_SEC,
    WINDOWING_VERSION,
    cue_time_from_row,
    extract_segment_baseline,
    iter_rest_sources_cue_before,
    lsl_to_sample,
    n_windows_3s_hop100,
    segment_to_3s_hop100_windows,
    slide_3s_from_interval,
    task_window_cue_0_to_4,
    wins_to_model,
)

# 兼容旧名
WIN_SEC_3S = WIN_SEC
HOP_SEC_3S = HOP_SEC
N_TIMES_3S = N_TIMES
_lsl_to_sample = lsl_to_sample
_wins_to_model = wins_to_model


def iter_rest_sources_from_table(
    rows: Sequence[Dict[str, Any]],
    t_lsl: np.ndarray,
    *,
    skip_rejected: bool = True,
    skip_invalid: bool = True,
    min_win_sec: float = WIN_SEC,
) -> List[Tuple[int, int, int]]:
    """从 trial_table 提取 Rest（label=0）源区间。返回 [(trial_id, t0, t1), ...]。

    兼容两种表形态：
    1) 独立 Rest 行：``phase=='rest'`` 且 ``label==0``（历史）
    2) 现行 v3：Cue 前静息挂在 L/R 试次行的 ``t_rest_start``/``t_rest_end``
       （语义即 Rest / label=0，不是单独「静息想象」试次行）
    """
    out: List[Tuple[int, int, int]] = []
    min_len = int(round(float(min_win_sec) * FS))
    for r in rows:
        if skip_rejected and str(r.get("rejected") or "0") == "1":
            continue
        if skip_invalid and str(r.get("invalid") or "0") == "1":
            continue
        t_s = r.get("t_rest_start")
        t_e = r.get("t_rest_end")
        try:
            if t_s is None or t_e is None or t_s == "" or t_e == "":
                continue
            ts = float(t_s)
            te = float(t_e)
            if ts != ts or te != te or te - ts < float(min_win_sec) - 1e-6:
                continue
        except (TypeError, ValueError):
            continue
        phase = str(r.get("phase") or "")
        lab = int(float(r.get("label") or -1)) if r.get("label") not in (None, "") else -1
        # 独立 Rest 行，或任意带 Cue 前静息打点的试次行
        is_rest_row = phase == "rest" and lab == 0
        if not is_rest_row and phase == "rest" and lab not in (0, -1):
            continue
        i0 = lsl_to_sample(t_lsl, ts)
        i1 = lsl_to_sample(t_lsl, te)
        if i1 - i0 < min_len:
            continue
        out.append((int(r.get("trial_id") or 0), i0, i1))
    return out


def cut_openbmi_align_from_table(
    x_filt: np.ndarray,
    t_lsl: np.ndarray,
    rows: List[Dict[str, Any]],
    *,
    task_phases: Optional[set] = None,
    include_rest_interval: bool = True,
    skip_invalid: bool = True,
    skip_rejected: bool = True,
) -> Tuple[List[np.ndarray], List[int], List[int], List[int]]:
    """从 trial_table 行切 OpenBMI 对齐窗。返回 wins, y_three, y_task, trial_ids。"""
    wins: List[np.ndarray] = []
    y_three: List[int] = []
    y_task: List[int] = []
    tids: List[int] = []

    cue_samples: List[int] = []
    n_left = n_right = 0

    for r in rows:
        if skip_rejected and str(r.get("rejected") or "0") == "1":
            continue
        if skip_invalid and str(r.get("invalid") or "0") == "1":
            continue
        phase = str(r.get("phase") or "")
        if task_phases is not None and phase and phase not in task_phases:
            continue
        lab = int(r.get("label") or -1)
        if lab not in (1, 2):
            continue
        t_cue = cue_time_from_row(r)
        if t_cue is None:
            continue
        cue_idx = lsl_to_sample(t_lsl, t_cue)
        seg = task_window_cue_0_to_4(x_filt, cue_idx, FS)
        if seg is None:
            continue
        seg_wins = segment_to_3s_hop100_windows(seg, FS, zscore=True)
        ws = wins_to_model(seg_wins)
        if not ws:
            continue
        tid = int(r["trial_id"])
        cue_samples.append(cue_idx)
        if lab == 1:
            n_left += 1
        else:
            n_right += 1
        for w in ws:
            wins.append(w)
            y_three.append(lab)
            y_task.append(1)
            tids.append(tid)

    if include_rest_interval:
        max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        rest_sources = iter_rest_sources_from_table(
            rows,
            t_lsl,
            skip_rejected=skip_rejected,
            skip_invalid=skip_invalid,
            min_win_sec=WIN_SEC,
        )
        if not rest_sources and cue_samples:
            fb = iter_rest_sources_cue_before(
                np.asarray(sorted(cue_samples), dtype=int),
                FS,
                x_filt.shape[0],
                rest_sec=4.0,
                task_sec=TASK_SEC,
                min_win_sec=WIN_SEC,
            )
            rest_sources = [(-(i + 1), int(t0), int(t1)) for i, (t0, t1) in enumerate(fb)]
        for ri, (tid, t0, t1) in enumerate(rest_sources[: int(max_rest)]):
            seg = extract_segment_baseline(x_filt, int(t0), int(t1), FS, baseline_sec=BASELINE_BEFORE_CUE_S)
            if seg is None:
                continue
            seg_wins = segment_to_3s_hop100_windows(seg, FS, zscore=True)
            ws = wins_to_model(seg_wins)
            for w in ws:
                wins.append(w)
                y_three.append(0)
                y_task.append(0)
                tids.append(int(tid) if tid > 0 else -(ri + 1))

    return wins, y_three, y_task, tids


__all__ = [
    "WINDOWING_VERSION",
    "FROZEN",
    "FS",
    "WIN_SEC",
    "WIN_SEC_3S",
    "HOP_SEC",
    "HOP_SEC_3S",
    "N_TIMES",
    "N_TIMES_3S",
    "TASK_SEC",
    "BASELINE_BEFORE_CUE_S",
    "cue_time_from_row",
    "lsl_to_sample",
    "_lsl_to_sample",
    "extract_segment_baseline",
    "task_window_cue_0_to_4",
    "segment_to_3s_hop100_windows",
    "n_windows_3s_hop100",
    "iter_rest_sources_cue_before",
    "iter_rest_sources_from_table",
    "wins_to_model",
    "_wins_to_model",
    "slide_3s_from_interval",
    "cut_openbmi_align_from_table",
]


if __name__ == "__main__":
    # 自检：不依赖 session 数据
    print("windowing_version=", WINDOWING_VERSION)
    print("ok", Path(__file__).name)
