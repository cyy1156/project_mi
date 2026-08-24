"""v2 在线推理服务（M2）：LSL 环形缓冲 → 3s 判定窗 → 在线预处理（复用 preprocess_lab）→ 双头推理 → 串行门控。

M2 验收命门：在线/离线一致性 —— filter_consistency_report()（见文件尾部）。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PREPROCESS_ROOT = str(Path(__file__).resolve().parents[2] / "code" / "preprocess_lab")
if _PREPROCESS_ROOT not in sys.path:
    sys.path.insert(0, _PREPROCESS_ROOT)

FS = 250.0
N_TIMES_3S = 750
CHANNEL_ORDER = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]  # 冻结序（索引 0–7）


class RingBuffer:
    """(T, 8) 环形缓冲 @250Hz；后台线程拉 LSL（或测试直接 push）。"""

    def __init__(self, n_ch: int = 8, capacity_s: float = 60.0, fs: float = FS):
        self.fs = fs
        self.n_ch = n_ch
        self.cap = int(capacity_s * fs)
        self._buf = np.zeros((self.cap, n_ch), dtype=np.float64)
        self._n = 0
        self._lock = threading.Lock()
        self._inlet = None
        self._thread: Optional[threading.Thread] = None
        self._stop = False

    # —— 数据源 ——
    def attach_lsl(self, stream_name: str = "OpenBCI_EEG") -> None:
        from pylsl import StreamInlet, resolve_byprop

        streams = resolve_byprop("name", stream_name, timeout=5.0)
        if not streams:
            raise RuntimeError(f"未找到 LSL 流 {stream_name}")
        self._inlet = StreamInlet(streams[0], max_buflen=int(self.cap / 80) + 2)
        self._thread = threading.Thread(target=self._pull_loop, daemon=True)
        self._thread.start()

    def _pull_loop(self) -> None:
        while not self._stop:
            sample, ts = self._inlet.pull_sample(timeout=0.1)
            if sample is not None:
                self.push(np.asarray(sample, dtype=np.float64)[: self.n_ch])

    def push(self, sample_tc: np.ndarray) -> None:
        with self._lock:
            n = len(sample_tc)
            pos = self._n % self.cap
            if pos + n <= self.cap:
                self._buf[pos : pos + n] = sample_tc
            else:
                k = self.cap - pos
                self._buf[pos:] = sample_tc[:k]
                self._buf[: n - k] = sample_tc[k:]
            self._n += n

    # —— 取数 ——
    def window_ending_at(self, t_end_lsl: float, n_samples: int = N_TIMES_3S,
                         t_now_lsl: Optional[float] = None) -> Optional[np.ndarray]:
        """取结束于 t_end_lsl 的 (n_samples, 8) 窗；数据未到 → None。"""
        from pylsl import local_clock

        now = local_clock() if t_now_lsl is None else t_now_lsl
        if now < t_end_lsl:
            return None  # 判定时刻未到（状态机已 wait，此为防御）
        with self._lock:
            end_idx = self._n  # 近似：缓冲尾 ≈ now（pull 延迟 <50ms 由验收测试量化）
            lag = max(0, int(round((now - t_end_lsl) * self.fs)))
            end = end_idx - lag
            start = end - n_samples
            if start < 0 or end > self._n:
                return None
            a, b = start % self.cap, end % self.cap
            if a < b:
                return self._buf[a:b].copy()
            return np.concatenate([self._buf[a:], self._buf[:b]], axis=0)

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class OnlinePreprocessor:
    """滚动尾段滤波：取尾 buffer_s 原始 → CAR → notch+bandpass(8–30) → 末 750 点 → 逐窗 z-score。

    与离线管线同函数（直接 import preprocess_lab），区别仅在滤波上下文长度
    （离线=整段连续；在线=尾 12s）——差异由 filter_consistency_report() 量化。
    """

    def __init__(self, buffer_s: float = 12.0, *, l_freq: float = 8.0, h_freq: float = 30.0):
        from src.common.steps.filter_car import car_reference, notch_and_bandpass
        from src.common.steps.resample_zscore import trial_zscore

        self._car = car_reference
        self._filt = notch_and_bandpass
        self._zs = trial_zscore
        self.buffer_s = buffer_s
        self.l_freq, self.h_freq = l_freq, h_freq

    def process(self, raw_tail_tc: np.ndarray) -> np.ndarray:
        """raw_tail_tc: (T,8) 原始尾段（T ≥ 750）→ (8,750) 模型输入。"""
        x = self._car(np.asarray(raw_tail_tc, dtype=np.float64))
        x = self._filt(x, FS, l_freq=self.l_freq, h_freq=self.h_freq)
        win_tc = x[-N_TIMES_3S:]
        return self._zs(win_tc).T.astype(np.float32)  # (8,750)


class InferenceService:
    """判定服务：judge(t_cue_lsl, t_rel) → {"pred","p_max","gated"}。

    registry: adapt_engine.ModelRegistry；readout 默认串行门控。
    """

    def __init__(self, buffer: RingBuffer, registry, pre: OnlinePreprocessor, *,
                 task_p_on: float = 0.6, tail_s: float = 12.0):
        self.buffer = buffer
        self.registry = registry
        self.pre = pre
        self.task_p_on = task_p_on
        self.tail_s = tail_s

    def judge(self, t_cue_lsl: float, t_rel: float) -> Optional[Dict]:
        from pylsl import local_clock

        from adapt_engine.readout import serial_gating

        t_end = t_cue_lsl + t_rel
        win_tc = self.buffer.window_ending_at(t_end, N_TIMES_3S, t_now_lsl=local_clock())
        if win_tc is None:
            return None
        tail_n = int(self.tail_s * FS)
        tail = self.buffer.window_ending_at(t_end, tail_n, t_now_lsl=local_clock())
        if tail is None:
            tail = win_tc
        window = self.pre.process(tail)
        heads = self.registry.forward_heads(window)
        p_task = heads["p_task"]
        p_three = heads["p_three"]
        out = serial_gating(p_task, p_three, task_p_on=self.task_p_on)
        out["window"] = window
        return out


# ——————————————————————————————
# M2 验收：在线/离线一致性
# ——————————————————————————————

def offline_windows(x_cont_tc: np.ndarray, cue_idx: int,
                    judgment_times=(3.0, 4.0, 5.0, 6.0)) -> np.ndarray:
    """离线路径：整段 CAR+notch+bandpass 后按 cue 切 3s 窗（结束于 cue+t）→ z-score → (n_j,8,750)。"""
    from src.common.steps.filter_car import car_reference, notch_and_bandpass
    from src.common.steps.resample_zscore import trial_zscore

    x = car_reference(np.asarray(x_cont_tc, dtype=np.float64))
    x = notch_and_bandpass(x, FS, l_freq=8.0, h_freq=30.0)
    outs = []
    for t in judgment_times:
        end = cue_idx + int(round(t * FS))
        w = trial_zscore(x[end - N_TIMES_3S : end]).T
        outs.append(w.astype(np.float32))
    return np.stack(outs)


def online_windows(x_cont_tc: np.ndarray, cue_idx: int, *, tail_s: float = 12.0,
                   judgment_times=(3.0, 4.0, 5.0, 6.0)) -> np.ndarray:
    """在线路径：每个判定点独立取尾段滤波（模拟 RingBuffer+OnlinePreprocessor 行为）。"""
    pre = OnlinePreprocessor(buffer_s=tail_s)
    outs = []
    for t in judgment_times:
        end = cue_idx + int(round(t * FS))
        tail = x_cont_tc[max(0, end - int(tail_s * FS)) : end]
        outs.append(pre.process(tail))
    return np.stack(outs)


def filter_consistency_report(x_cont_tc: np.ndarray, cue_idx: int) -> Dict:
    """量化在线/离线预处理差异（M2 验收）。判据建议：argmax 不一致率 <1%、max|Δz| 报告值。"""
    off = offline_windows(x_cont_tc, cue_idx)
    on = online_windows(x_cont_tc, cue_idx)
    diff = np.abs(off - on)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "per_window_max": [float(d.max()) for d in diff],
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 20, (int(40 * FS), 8))  # 40s 合成连续段
    cue = int(10 * FS)
    rep = filter_consistency_report(x, cue)
    print("一致性报告（合成数据）：")
    for k, v in rep.items():
        print(f"  {k}: {v if not isinstance(v, list) else [round(f, 4) for f in v]}")
