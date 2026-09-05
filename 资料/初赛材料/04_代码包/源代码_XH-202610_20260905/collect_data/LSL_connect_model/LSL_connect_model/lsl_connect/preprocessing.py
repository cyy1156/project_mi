"""
第 5 课：EEG / 加速度预处理（缩放 + 可选滤波）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes
from scipy.signal import butter, sosfilt, sosfilt_zi

# Cyton：ADC 计数 -> 微伏 (uV)
SCALE_EEG = 4_500_000 / 24 / (2**23 - 1)

# 加速度计原始值 -> m/s^2（按 BrainFlow / OpenBCI 常用换算）
SCALE_ACCEL = 0.002 / (2**4)


@dataclass
class PreprocessConfig:
    """预处理开关与滤波参数。"""

    sample_rate: int = 250
    filter_enabled: bool = True
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 45.0
    notch_enabled: bool = True
    notch_low_hz: float = 49.0
    notch_high_hz: float = 51.0
    filter_order: int = 2


class StreamingEegFilter:
    """因果流式带通 + 可选陷波（实时采集用）。

    取代旧的「按批（≤25 样本）零相位滤波」：零相位需要前后文，逐批独立做会在每批
    两端算错、批间留下接缝伪迹。本类为每通道维护滤波状态（zi），新批次接着上一批
    状态继续滤，等价于对整段连续信号做因果滤波——无块边界伪迹、延迟极小。代价是
    引入相位延迟（非零相位），但 MI 解码读 μ/β 功率/包络，不受影响。
    """

    def __init__(
        self,
        sample_rate: int = 250,
        bandpass_low_hz: float = 0.5,
        bandpass_high_hz: float = 45.0,
        notch_enabled: bool = True,
        notch_low_hz: float = 49.0,
        notch_high_hz: float = 51.0,
        filter_order: int = 2,
    ) -> None:
        bp = butter(
            filter_order,
            [bandpass_low_hz, bandpass_high_hz],
            btype="band",
            fs=sample_rate,
            output="sos",
        )
        if notch_enabled:
            bs = butter(
                filter_order,
                [notch_low_hz, notch_high_hz],
                btype="bandstop",
                fs=sample_rate,
                output="sos",
            )
            self._sos = np.vstack([bp, bs])
        else:
            self._sos = np.asarray(bp)
        # 单位阶跃响应的初始条件，用于按首样本播种、抑制启动瞬态
        self._zi_unit = sosfilt_zi(self._sos)
        self.reset()

    def reset(self) -> None:
        """清空滤波状态；断开/重连板卡后调用以重新起算。"""
        self._zi: np.ndarray | None = None
        self._n_ch: int | None = None

    def process(self, eeg_uv: np.ndarray) -> np.ndarray:
        """eeg_uv: (n_channels, n_samples) µV；返回同形状 float64。

        首次调用按各通道首样本播种状态（稳态种子），之后逐批衔接。
        """
        x = np.asarray(eeg_uv, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] == 0:
            return x
        n_ch = x.shape[0]
        if self._zi is None or self._n_ch != n_ch:
            self._n_ch = n_ch
            self._zi = np.stack([self._zi_unit * x[c, 0] for c in range(n_ch)])
        out = np.empty_like(x)
        for c in range(n_ch):
            out[c], self._zi[c] = sosfilt(self._sos, x[c], zi=self._zi[c])
        return out

    @classmethod
    def from_config(cls, config: "PreprocessConfig") -> "StreamingEegFilter":
        return cls(
            sample_rate=config.sample_rate,
            bandpass_low_hz=config.bandpass_low_hz,
            bandpass_high_hz=config.bandpass_high_hz,
            notch_enabled=bool(getattr(config, "notch_enabled", True)),
            notch_low_hz=config.notch_low_hz,
            notch_high_hz=config.notch_high_hz,
            filter_order=config.filter_order,
        )


def counts_to_microvolts(eeg_counts: np.ndarray) -> np.ndarray:
    """将 EEG 原始计数转为微伏。输入/输出形状: (n_channels, n_samples)。"""
    return eeg_counts.astype(np.float64) * SCALE_EEG


def counts_to_accel_ms2(accel_counts: np.ndarray) -> np.ndarray:
    """将加速度计原始计数转为 m/s^2。形状: (3, n_samples)。"""
    return accel_counts.astype(np.float64) * SCALE_ACCEL


def apply_eeg_filters(
    eeg_uv: np.ndarray,
    config: PreprocessConfig | None = None,
) -> None:
    """
    原地滤波。eeg_uv 形状必须为 (n_channels, n_samples)。
    """
    if config is None:
        config = PreprocessConfig()
    if not config.filter_enabled:
        return

    n_ch, _ = eeg_uv.shape
    for ch in range(n_ch):
        DataFilter.perform_bandpass(
            eeg_uv[ch],
            config.sample_rate,
            config.bandpass_low_hz,
            config.bandpass_high_hz,
            config.filter_order,
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            0,
        )
        if getattr(config, "notch_enabled", True):
            DataFilter.perform_bandstop(
                eeg_uv[ch],
                config.sample_rate,
                config.notch_low_hz,
                config.notch_high_hz,
                config.filter_order,
                FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                0,
            )


def preprocess_eeg_batch(
    eeg_counts: np.ndarray,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """
    计数 -> 微伏 -> 可选滤波，返回 float32 数组 (n_channels, n_samples)。
    """
    if config is None:
        config = PreprocessConfig()
    eeg_uv = counts_to_microvolts(eeg_counts)
    apply_eeg_filters(eeg_uv, config)
    return eeg_uv.astype(np.float32)


def preprocess_accel_batch(accel_counts: np.ndarray) -> np.ndarray:
    """加速度计数 -> m/s^2，返回 float32 (3, n_samples)。"""
    return counts_to_accel_ms2(accel_counts).astype(np.float32)