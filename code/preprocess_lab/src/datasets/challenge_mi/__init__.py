"""挑战杯官方指定集 · 预处理（Exp34）。

协议：先切 3s trial → 选通道 → CAR → notch50 → bandpass 8–30 → 窗内 z-score。
输出 shape: (N, 1, C, 750)，C∈{59,8}。
"""

from __future__ import annotations

__all__ = ["PROTOCOL_59", "PROTOCOL_8"]

PROTOCOL_59 = "challenge_mi_3s_59ch"
PROTOCOL_8 = "challenge_mi_3s_8ch"
