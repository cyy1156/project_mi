"""3 s / 100 ms 重叠滑窗（实验 20 · OpenBMI Acc_paper 旁路）。

与 slide_2s_hop100 同协议，仅 win_sec=3.0 → n_times=750 @ 250 Hz。
禁止改坏 2s 默认入口。
"""
from __future__ import annotations

from src.common.steps.slide_1s import (  # noqa: F401
    FS_OUT,
    extract_segment_baseline,
    n_windows_for_duration,
    segment_to_windows,
)

WIN_SEC = 3.0
HOP_SEC = 0.1  # 100 ms
N_TIMES_3S = int(round(WIN_SEC * FS_OUT))  # 750
HOP_SAMPLES = int(round(HOP_SEC * FS_OUT))  # 25


def segment_to_3s_hop100_windows(seg_tc, fs_in, *, fs_out: float = FS_OUT, zscore: bool = True):
    return segment_to_windows(
        seg_tc,
        fs_in,
        win_sec=WIN_SEC,
        hop_sec=HOP_SEC,
        fs_out=fs_out,
        zscore=zscore,
    )


def n_windows_3s_hop100(duration_sec: float) -> int:
    return n_windows_for_duration(duration_sec, win_sec=WIN_SEC, hop_sec=HOP_SEC)
