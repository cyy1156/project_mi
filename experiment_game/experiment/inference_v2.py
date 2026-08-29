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
WIN_SEC_3S = 3.0
HOP_SEC_100 = 0.1
BASELINE_BEFORE_CUE_S = 0.5
MI_TASK_SEC = 4.0
# 真机：连续无新样本超过该秒数 → 判定 EEG 断流（勿再用陈旧窗推理）
EEG_STALE_TIMEOUT_S = 3.0
# judge 软陈旧（总册 §5.2）：缓冲年龄 >1s → stale=true，供计分排除
JUDGE_BUF_STALE_S = 1.0

from experiment_game.core.channel_layout import (  # noqa: E402
    CHANNEL_ORDER,
    permute_ch_time_to_model,
)

# 全局与设备/采集序一致；模型 forward 前 permute_ch_time_to_model


def readout_from_heads(
    p_task: Optional[np.ndarray],
    p_three: np.ndarray,
    *,
    task_p_on: float,
) -> Dict:
    """Task 头缺失（E1f）或 task_p_on≤0 时跳过串行门控，直接 three 头 argmax。"""
    from adapt_engine.readout import serial_gating

    p3 = np.asarray(p_three, dtype=np.float64).ravel()
    if p_task is None or float(task_p_on) <= 0.0:
        pred = int(np.argmax(p3))
        return {"pred": pred, "p_max": float(np.max(p3)), "gated": False}
    return serial_gating(p_task, p_three, task_p_on=float(task_p_on))


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
        # 仿真回放：1 LSL 秒灌入 fs×scale 样本（默认 1=实时 LSL）
        self.push_rate_scale = 1.0
        # 最近一次 push 的单调时钟（秒）；None=尚未收到任何样本
        self._last_push_mono: Optional[float] = None
        # 看门狗起点：无样本时按「自创建/挂流起」计时（避免永不 push 时永远不报警）
        self._watch_mono: float = time.monotonic()
        # 可选 EEGBus（runtime）；push 时 note_push
        self._bus = None

    def attach_bus(self, bus) -> None:
        """挂接 runtime.EEGBus，使 push 同步更新总线健康。"""
        self._bus = bus

    # —— 数据源 ——
    def attach_lsl(self, stream_name: str = "OpenBCI_EEG", *, timeout_s: float = 5.0) -> None:
        from pylsl import StreamInlet, resolve_byprop

        streams = resolve_byprop("name", stream_name, timeout=float(timeout_s))
        if not streams:
            raise RuntimeError(f"未找到 LSL 流 {stream_name}（timeout={timeout_s}s）")
        self._inlet = StreamInlet(streams[0], max_buflen=int(self.cap / 80) + 2)
        self._watch_mono = time.monotonic()
        self._thread = threading.Thread(target=self._pull_loop, daemon=True)
        self._thread.start()

    def _pull_loop(self) -> None:
        while not self._stop:
            sample, ts = self._inlet.pull_sample(timeout=0.1)
            if sample is not None:
                row = np.asarray(sample, dtype=np.float64)[: self.n_ch]
                t_lsl = None
                if ts is not None:
                    t_lsl = np.asarray([float(ts)], dtype=np.float64)
                self.push(row.reshape(1, -1), t_lsl=t_lsl)

    def push(self, sample_tc: np.ndarray, t_lsl: Optional[np.ndarray] = None) -> None:
        sample_tc = np.asarray(sample_tc, dtype=np.float64)
        if sample_tc.ndim == 1:
            # 单样本 (C,) → (1, C)；否则 len()=C 会把一行广播成 C 行相同数据
            sample_tc = sample_tc.reshape(1, -1)
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
            self._last_push_mono = time.monotonic()
        bus = self._bus
        if bus is not None:
            try:
                if t_lsl is not None:
                    # 扇出订户（CSV 等）；count=True 更新健康（替代单独 note_push）
                    bus.publish(np.asarray(t_lsl, dtype=np.float64), sample_tc, count=True)
                else:
                    bus.note_push(n)
            except Exception:  # noqa: BLE001
                pass

    def last_push_age_s(self) -> Optional[float]:
        """距最近一次 push 的秒数；尚无样本时返回自 _watch_mono 起的秒数。"""
        with self._lock:
            ts = self._last_push_mono
            n = self._n
            watch = self._watch_mono
        now = time.monotonic()
        if ts is not None and n > 0:
            return max(0.0, now - ts)
        return max(0.0, now - watch)

    def is_stale(self, timeout_s: float = EEG_STALE_TIMEOUT_S) -> bool:
        """超过 timeout_s 无新样本（含从未收到样本）。"""
        age = self.last_push_age_s()
        if age is None:
            return False
        return age > float(timeout_s)

    def stale_status(
        self, timeout_s: float = EEG_STALE_TIMEOUT_S
    ) -> Optional[Dict[str, float]]:
        """断流时返回 {age_s, timeout_s, n_samples}；否则 None。"""
        age = self.last_push_age_s()
        if age is None or age <= float(timeout_s):
            return None
        with self._lock:
            n = int(self._n)
        return {"age_s": float(age), "timeout_s": float(timeout_s), "n_samples": n}

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
            lag = max(0, int(round((now - t_end_lsl) * self.fs * self.push_rate_scale)))
            end = end_idx - lag
            start = end - n_samples
            if start < 0 or end > self._n:
                return None
            a, b = start % self.cap, end % self.cap
            if a < b:
                return self._buf[a:b].copy()
            return np.concatenate([self._buf[a:], self._buf[:b]], axis=0)

    def snapshot_tail(self, seg_s: float, t_now_lsl: Optional[float] = None) -> Optional[np.ndarray]:
        """取缓冲尾 seg_s 秒原始段 (T,8)。"""
        from pylsl import local_clock

        now = local_clock() if t_now_lsl is None else t_now_lsl
        n = max(1, int(round(seg_s * self.fs)))
        return self.window_ending_at(now, n, t_now_lsl=now)

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

    def process(self, raw_tail_tc: np.ndarray, *, zscore: bool = True) -> np.ndarray:
        """raw_tail_tc: (T,8) 原始尾段（T ≥ 750）→ (8,750) 模型输入。"""
        x = self._car(np.asarray(raw_tail_tc, dtype=np.float64))
        x = self._filt(x, FS, l_freq=self.l_freq, h_freq=self.h_freq)
        win_tc = x[-N_TIMES_3S:]
        if zscore:
            return self._zs(win_tc).T.astype(np.float32)  # (8,750)
        return win_tc.astype(np.float32)

    def process_segment(self, raw_tc: np.ndarray) -> np.ndarray:
        """整段 CAR+notch+bp(8–30)，不做 z-score；供 v3 特征分析。"""
        x = self._car(np.asarray(raw_tc, dtype=np.float64))
        return self._filt(x, FS, l_freq=self.l_freq, h_freq=self.h_freq)

    def process_openbmi_task_window(
        self,
        raw_tail_tc: np.ndarray,
        *,
        seg_len_s: float,
        win_start_rel: float,
        win_end_rel: float,
        baseline_sec: float = BASELINE_BEFORE_CUE_S,
        zscore: bool = True,
    ) -> Optional[np.ndarray]:
        """OpenBMI 对齐：段 [Cue−baseline, Cue+win_end] 滤波后减 Cue 前基线，取 3s 窗 z-score。

        raw_tail_tc 须覆盖整段且尾样本对齐 Cue+win_end_rel。
        win_start_rel / win_end_rel 相对 Cue（秒）。
        """
        seg_n = int(round(seg_len_s * FS))
        if raw_tail_tc.shape[0] < seg_n:
            return None
        x = self._car(np.asarray(raw_tail_tc, dtype=np.float64))
        x = self._filt(x, FS, l_freq=self.l_freq, h_freq=self.h_freq)
        x_seg = x[-seg_n:]
        n_base = int(round(baseline_sec * FS))
        if n_base <= 0 or x_seg.shape[0] < n_base + N_TIMES_3S:
            return None
        i0 = n_base + int(round(win_start_rel * FS))
        i1 = n_base + int(round(win_end_rel * FS))
        if i1 - i0 != N_TIMES_3S:
            i1 = i0 + N_TIMES_3S
        if i0 < n_base or i1 > x_seg.shape[0]:
            return None
        base = x_seg[:n_base].mean(axis=0, keepdims=True)
        win_tc = x_seg[i0:i1] - base
        if zscore:
            return self._zs(win_tc).T.astype(np.float32)
        return win_tc.T.astype(np.float32)


class InferenceService:
    """判定服务：judge(t_cue_lsl, t_rel) → {"pred","p_max","gated"} 或 signal_bad。

    registry: adapt_engine.ModelRegistry；readout 默认串行门控。
    """

    def __init__(
        self,
        buffer: RingBuffer,
        registry,
        pre: OnlinePreprocessor,
        *,
        task_p_on: float = 0.6,
        tail_s: float = 12.0,
        signal_quality=None,
        forward_window: bool = False,
        window_mode: str = "legacy",
        mi_task_sec: float = MI_TASK_SEC,
        baseline_before_cue_s: float = BASELINE_BEFORE_CUE_S,
        lsl_eeg_scale: float = 1.0,
    ):
        self.buffer = buffer
        self.registry = registry
        self.pre = pre
        self.task_p_on = task_p_on
        self.tail_s = tail_s
        self.signal_quality = signal_quality
        self.forward_window = forward_window
        self.window_mode = window_mode
        self.mi_task_sec = float(mi_task_sec)
        self.baseline_before_cue_s = float(baseline_before_cue_s)
        # 仿真回放：墙钟 LSL 推进 1/speed，EEG 内容推进 1s；取窗须缩放
        self.lsl_eeg_scale = max(1e-9, float(lsl_eeg_scale))
        # False：跳过断流检测（仿真回放 gap 不应记 signal_bad）
        self.stale_check_enabled = True

    def _lsl_at_eeg_rel(self, t_cue_lsl: float, eeg_rel_s: float) -> float:
        return float(t_cue_lsl) + float(eeg_rel_s) * self.lsl_eeg_scale

    def _annotate_buf_age(self, out: Dict) -> Dict:
        """附加 buf_age_s；年龄 > JUDGE_BUF_STALE_S 时标 stale=true（软陈旧）。"""
        age = self.buffer.last_push_age_s()
        age_f = float(age) if age is not None else 0.0
        out["buf_age_s"] = age_f
        if age_f > float(JUDGE_BUF_STALE_S):
            out["stale"] = True
        else:
            out.setdefault("stale", False)
        return out

    def judge(self, t_cue_lsl: float, t_rel: float) -> Optional[Dict]:
        if self.stale_check_enabled:
            stale = self.buffer.stale_status()
            if stale is not None:
                return self._annotate_buf_age(
                    {
                        "eeg_stale": True,
                        "signal_bad": True,
                        "reason": "eeg_stale",
                        "age_s": stale["age_s"],
                        "timeout_s": stale["timeout_s"],
                        "n_samples": stale["n_samples"],
                        "t_rel": float(t_rel),
                    }
                )
        if self.window_mode == "openbmi_hop100":
            out = self.judge_openbmi_hop100(t_cue_lsl, t_rel)
            return self._annotate_buf_age(out) if out is not None else None
        out = self._judge_legacy(t_cue_lsl, t_rel)
        return self._annotate_buf_age(out) if out is not None else None

    def judge_openbmi_hop100(self, t_cue_lsl: float, t_win_end_rel: float) -> Optional[Dict]:
        """OpenBMI 3s/hop100：t_win_end_rel 为窗尾（相对 Cue）；窗 [end−3, end] ⊆ [0, MI]。"""
        from pylsl import local_clock

        from experiment_game.experiment.signal_quality import assess_eeg_window

        win_end = float(t_win_end_rel)
        win_start = win_end - WIN_SEC_3S
        if win_start < -1e-9 or win_end > self.mi_task_sec + 1e-9:
            return None

        t_seg_end = self._lsl_at_eeg_rel(t_cue_lsl, win_end)
        seg_len_s = self.baseline_before_cue_s + win_end
        now = local_clock()

        # OpenBMI 段级取窗：尾段长度 = Cue 前基线 + MI 窗尾（+0.5s 滤波过渡）
        tail_s = seg_len_s + 0.5
        tail_n = int(round(tail_s * FS))
        tail_raw = self.buffer.window_ending_at(t_seg_end, tail_n, t_now_lsl=now)
        if tail_raw is None:
            return None

        window = self.pre.process_openbmi_task_window(
            tail_raw,
            seg_len_s=seg_len_s,
            win_start_rel=win_start,
            win_end_rel=win_end,
            baseline_sec=self.baseline_before_cue_s,
        )
        if window is None:
            return None

        if self.signal_quality is not None and getattr(self.signal_quality, "enabled", True):
            qa = assess_eeg_window(window.T, self.signal_quality)
            if not qa["ok"]:
                return {
                    "signal_bad": True,
                    "reason": qa["reason"],
                    "signal_metrics": qa.get("metrics") or {},
                    "t_rel": win_end,
                    "win_start_rel": win_start,
                    "win_end_rel": win_end,
                }

        window = permute_ch_time_to_model(window)
        heads = self.registry.forward_heads(window)
        p_task = heads["p_task"]
        p_three = heads["p_three"]
        out = readout_from_heads(p_task, p_three, task_p_on=self.task_p_on)
        out["window"] = window
        out["p_three"] = [float(x) for x in np.asarray(p_three).ravel()]
        if p_task is not None:
            out["p_task"] = [float(x) for x in np.asarray(p_task).ravel()]
        out["t_rel"] = win_end
        out["win_start_rel"] = win_start
        out["win_end_rel"] = win_end
        pred = int(out["pred"])
        if pred < len(out["p_three"]):
            others = [out["p_three"][i] for i in range(len(out["p_three"])) if i != pred]
            out["margin"] = float(out["p_three"][pred] - max(others or [0.0]))
        else:
            out["margin"] = float(out.get("p_max", 0.0))
        return out

    def _judge_legacy(self, t_cue_lsl: float, t_rel: float) -> Optional[Dict]:
        from pylsl import local_clock

        from experiment_game.experiment.signal_quality import assess_eeg_window

        t_end = self._lsl_at_eeg_rel(t_cue_lsl, t_rel)
        if self.forward_window:
            t_end += (N_TIMES_3S / FS) * self.lsl_eeg_scale
        win_tc = self.buffer.window_ending_at(t_end, N_TIMES_3S, t_now_lsl=local_clock())
        if win_tc is None:
            return None

        # signal_quality.enabled=False 时跳过：所有窗进入模型，不因质量剔除
        if self.signal_quality is not None and getattr(self.signal_quality, "enabled", True):
            qa = assess_eeg_window(win_tc, self.signal_quality)
            if not qa["ok"]:
                return {
                    "signal_bad": True,
                    "reason": qa["reason"],
                    "signal_metrics": qa.get("metrics") or {},
                    "t_rel": t_rel,
                }

        tail_n = int(self.tail_s * FS)
        tail = self.buffer.window_ending_at(t_end, tail_n, t_now_lsl=local_clock())
        if tail is None:
            tail = win_tc
        window = self.pre.process(tail)
        window = permute_ch_time_to_model(window)
        heads = self.registry.forward_heads(window)
        p_task = heads["p_task"]
        p_three = heads["p_three"]
        out = readout_from_heads(p_task, p_three, task_p_on=self.task_p_on)
        out["window"] = window
        out["p_three"] = [float(x) for x in np.asarray(p_three).ravel()]
        if p_task is not None:
            out["p_task"] = [float(x) for x in np.asarray(p_task).ravel()]
        pred = int(out["pred"])
        if pred < len(out["p_three"]):
            others = [out["p_three"][i] for i in range(len(out["p_three"])) if i != pred]
            out["margin"] = float(out["p_three"][pred] - max(others or [0.0]))
        else:
            out["margin"] = float(out.get("p_max", 0.0))
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
