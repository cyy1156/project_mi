"""v3 实时脑电帧发布器：RingBuffer → ws eeg_frame。"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from experiment_game.experiment.feature_probe import bandpowers_fft
from experiment_game.experiment.inference_v2 import CHANNEL_ORDER, FS
from experiment_game.experiment.signal_quality import assess_eeg_window

if TYPE_CHECKING:
    from experiment_game.experiment.inference_v2 import RingBuffer, OnlinePreprocessor
    from experiment_game.experiment.v3_config import V3Config
    from experiment_game.experiment.ws_bridge import WsBridge


class EegFramePublisher:
    def __init__(
        self,
        buffer: "RingBuffer",
        bridge: "WsBridge",
        cfg: "V3Config",
        pre: Optional["OnlinePreprocessor"] = None,
        on_console: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.buffer = buffer
        self.bridge = bridge
        self.cfg = cfg
        self.pre = pre
        self.on_console = on_console
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._standards = cfg.standards()
        self._consec_fail = 0
        self._last_fail_log = 0.0
        self._warned = False
        self._sq_cfg = cfg.signal_quality_config()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        from pylsl import local_clock

        interval = float(self.cfg.eeg_frame_interval_s)
        win_s = float(self.cfg.eeg_frame_window_s)
        while not self._stop:
            t0 = time.monotonic()
            try:
                now = local_clock()
                raw = self.buffer.snapshot_tail(win_s, t_now_lsl=now)
                if raw is not None and raw.shape[0] >= 8:
                    n = raw.shape[0] - (raw.shape[0] % 4)
                    if n >= 4:
                        disp = raw[:n].reshape(-1, 4, raw.shape[1]).mean(axis=1)
                    else:
                        disp = raw
                    filt = raw
                    if self.pre is not None:
                        filt = self.pre.process_segment(raw)
                    mu, bl, _ = bandpowers_fft(filt, FS, self._standards)
                    qa = assess_eeg_window(raw, self._sq_cfg)
                    self.bridge.broadcast({
                        "type": "eeg_frame",
                        "t": now,
                        "fs_disp": FS / 4.0,
                        "ch": list(CHANNEL_ORDER),
                        "data": disp.T.tolist(),
                        "power_mu": mu.tolist(),
                        "power_beta": bl.tolist(),
                        "signal_ok": bool(qa["ok"]),
                        "signal_reason": qa.get("reason"),
                        "dead_channel_idx": (qa.get("metrics") or {}).get("dead_channel_idx"),
                    })
                self._consec_fail = 0
                self._warned = False
            except Exception as exc:
                self._consec_fail += 1
                now_m = time.monotonic()
                if now_m - self._last_fail_log >= 10.0:
                    self._last_fail_log = now_m
                    if self.on_console is not None:
                        self.on_console(
                            f"[v3] EEG 帧发布失败（连续 {self._consec_fail} 次）：{exc!r}"
                        )
                if not self._warned and self._consec_fail >= 10:
                    self._warned = True
                    try:
                        self.bridge.broadcast({
                            "type": "v3_warn",
                            "message": "EEG 帧流中断：实时脑电画面可能停更，请检查采集链路",
                        })
                    except Exception:
                        pass
            elapsed = time.monotonic() - t0
            time.sleep(max(0.01, interval - elapsed))
