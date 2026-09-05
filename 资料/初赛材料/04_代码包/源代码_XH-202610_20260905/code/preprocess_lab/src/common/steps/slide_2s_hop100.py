"""2 s / 100 ms 重叠滑窗（旁路协议；不与论文 1s/40ms 对齐）。"""
from __future__ import annotations

from src.common.steps.slide_1s import (  # noqa: F401
    FS_OUT,
    extract_segment_baseline,
    iter_rest_sources_cue_before,
    n_windows_for_duration,
    segment_to_windows,
    slide_windows,
)

WIN_SEC = 2.0
HOP_SEC = 0.1  # 100 ms
N_TIMES_2S = int(round(WIN_SEC * FS_OUT))  # 500
HOP_SAMPLES = int(round(HOP_SEC * FS_OUT))  # 25


def segment_to_2s_hop100_windows(seg_tc, fs_in, *, fs_out: float = FS_OUT, zscore: bool = True):
    return segment_to_windows(
        seg_tc,
        fs_in,
        win_sec=WIN_SEC,
        hop_sec=HOP_SEC,
        fs_out=fs_out,
        zscore=zscore,
    )


def n_windows_2s_hop100(duration_sec: float) -> int:
    return n_windows_for_duration(duration_sec, win_sec=WIN_SEC, hop_sec=HOP_SEC)
