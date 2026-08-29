"""BCI2a 连续 EEG 回放 → RingBuffer + eeg.csv。"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from experiment_game.experiment.channel_layout import (
    DEVICE_CHANNEL_LABELS,
    reorder_model_input_to_device,
)
from experiment_game.experiment.inference_v2 import FS, RingBuffer
from experiment_game.experiment.sim.run_to_session_map import SimTrialScript


def _extract_segment(x8: np.ndarray, start: int, end: int, n_target: int) -> np.ndarray:
    """从 mat 取 [start,end)，不足则 edge pad 到 n_target。"""
    start = max(0, int(start))
    end = min(int(x8.shape[0]), int(end))
    seg = x8[start:end]
    if len(seg) == 0:
        seg = np.zeros((1, x8.shape[1]), dtype=np.float64)
    if len(seg) >= n_target:
        return seg[:n_target].copy()
    pad = np.tile(seg[-1:], (n_target - len(seg), 1))
    return np.concatenate([seg, pad], axis=0)


def build_schedule_align_timeline(
    script: SimTrialScript,
    *,
    rest_s: float = 4.0,
    prep_s: float = 2.0,
    mi_s: float = 4.0,
    iti_s: float = 3.0,
) -> np.ndarray:
    """拼接整场 session 的 (T, 8) @250Hz（schedule_align）。"""
    fs = script.fs
    n_rest = int(round(rest_s * fs))
    n_prep = int(round(prep_s * fs))
    n_mi = int(round(mi_s * fs))
    n_iti = int(round(iti_s * fs))
    chunks: List[np.ndarray] = []
    x8 = script.x8

    for tr in script.trials:
        cue = tr.cue_sample
        if int(tr.label) == 0:
            rs = tr.rest_start_sample
            prep = _extract_segment(x8, rs - n_prep - n_rest, rs - n_rest, n_prep)
            mi = _extract_segment(x8, rs, rs + n_mi, n_mi)
            iti = _extract_segment(x8, cue, cue + n_iti, n_iti)
            # Rest 试次：范式跳过 inter_trial_rest，回放也不推首段 4s rest
            chunks.extend([prep, mi, iti])
        else:
            rest = _extract_segment(x8, tr.rest_start_sample, cue, n_rest)
            prep = _extract_segment(x8, cue - n_prep - n_rest, cue - n_rest, n_prep)
            mi = _extract_segment(x8, cue, cue + n_mi, n_mi)
            iti = _extract_segment(x8, cue + n_mi, cue + n_mi + n_iti, n_iti)
            chunks.extend([rest, prep, mi, iti])

    if not chunks:
        return np.zeros((0, 8), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def build_timing_align_timeline(
    script: SimTrialScript,
    *,
    rest_s: float = 4.0,
    prep_s: float = 2.0,
    mi_s: float = 4.0,
    iti_s: float = 3.0,
) -> np.ndarray:
    """timing_align：ITI 段尽量使用 mat 试次间真实 EEG（仍按 trial 顺序拼接）。"""
    fs = script.fs
    n_rest = int(round(rest_s * fs))
    n_prep = int(round(prep_s * fs))
    n_mi = int(round(mi_s * fs))
    n_iti_min = int(round(iti_s * fs))
    chunks: List[np.ndarray] = []
    x8 = script.x8
    cues = [tr.cue_sample for tr in script.trials]

    for i, tr in enumerate(script.trials):
        cue = tr.cue_sample
        if i + 1 < len(cues):
            next_cue = cues[i + 1]
            gap_end = min(cue + n_mi + n_iti_min, next_cue - n_rest)
            gap_end = max(gap_end, cue + n_mi)
            iti = _extract_segment(x8, cue + n_mi, gap_end, max(n_iti_min, gap_end - cue - n_mi))
        else:
            iti = _extract_segment(x8, cue + n_mi, cue + n_mi + n_iti_min, n_iti_min)
        if int(tr.label) == 0:
            rs = tr.rest_start_sample
            prep = _extract_segment(x8, rs - n_prep - n_rest, rs - n_rest, n_prep)
            mi = _extract_segment(x8, rs, rs + n_mi, n_mi)
        else:
            rest = _extract_segment(x8, tr.rest_start_sample, cue, n_rest)
            prep = _extract_segment(x8, cue - n_prep - n_rest, cue - n_rest, n_prep)
            mi = _extract_segment(x8, cue, cue + n_mi, n_mi)
        chunks.extend([prep, mi, iti] if int(tr.label) == 0 else [rest, prep, mi, iti])

    if not chunks:
        return np.zeros((0, 8), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def build_replay_timeline(
    script: SimTrialScript,
    *,
    align_mode: str = "schedule_align",
    rest_s: float = 4.0,
    prep_s: float = 2.0,
    mi_s: float = 4.0,
    iti_s: float = 3.0,
) -> np.ndarray:
    if str(align_mode).lower() == "timing_align":
        return build_timing_align_timeline(
            script, rest_s=rest_s, prep_s=prep_s, mi_s=mi_s, iti_s=iti_s
        )
    return build_schedule_align_timeline(
        script, rest_s=rest_s, prep_s=prep_s, mi_s=mi_s, iti_s=iti_s
    )


class Bci2aReplaySource:
    """按墙钟向 RingBuffer 灌样本；可选写 eeg.csv。"""

    def __init__(
        self,
        script: SimTrialScript,
        buf: RingBuffer,
        *,
        eeg_csv_path: Optional[Path] = None,
        speed: float = 1.0,
        align_mode: str = "schedule_align",
        rest_s: float = 4.0,
        prep_s: float = 2.0,
        mi_s: float = 4.0,
        iti_s: float = 3.0,
    ):
        self.script = script
        self.buf = buf
        self.eeg_csv_path = Path(eeg_csv_path) if eeg_csv_path else None
        self.speed = max(0.01, float(speed))
        self.align_mode = str(align_mode)
        self.timeline = build_replay_timeline(
            script,
            align_mode=align_mode,
            rest_s=rest_s,
            prep_s=prep_s,
            mi_s=mi_s,
            iti_s=iti_s,
        )
        self.fs = float(script.fs or FS)
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._t0: Optional[float] = None
        self._csv_file = None
        self._csv_writer = None
        self.samples_pushed = 0

    @property
    def ring_buffer(self) -> RingBuffer:
        return self.buf

    def start(self) -> None:
        from pylsl import local_clock

        if self.eeg_csv_path is not None:
            self.eeg_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = self.eeg_csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["lsl_time"] + list(DEVICE_CHANNEL_LABELS))

        self._t0 = local_clock()
        self._stop = False
        self._thread = threading.Thread(target=self._feed_loop, daemon=True, name="Bci2aReplay")
        self._thread.start()

    def _feed_loop(self) -> None:
        from pylsl import local_clock

        n = len(self.timeline)
        i = 0
        while not self._stop and i < n:
            now = local_clock()
            assert self._t0 is not None
            target_i = int((now - self._t0) * self.fs * self.speed)
            while i <= target_i and i < n:
                row = self.timeline[i : i + 1]
                row_dev = reorder_model_input_to_device(row)
                self.buf.push(row_dev)
                lsl_t = self._t0 + i / (self.fs * self.speed)
                if self._csv_writer is not None:
                    self._csv_writer.writerow(
                        [f"{lsl_t:.6f}"] + [f"{v:.6f}" for v in row_dev[0]]
                    )
                i += 1
                self.samples_pushed += 1
            time.sleep(0.002)
        if self._csv_file is not None:
            self._csv_file.flush()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

    def session_duration_s(self) -> float:
        return len(self.timeline) / self.fs / self.speed
