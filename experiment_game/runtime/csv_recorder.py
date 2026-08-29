"""EEGBus 订阅者：将样本原子追写成 eeg.csv（总册 W2 落盘订户）。

仿真回放与真机 LiveEegCapture 共用；替代 lsl_connect Recorder 双轨写盘。
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from experiment_game.core.atomic_io import atomic_write_json
from experiment_game.core.channel_layout import DEVICE_CHANNEL_LABELS


class CsvRecorderSubscriber:
    """``on_chunk(t_lsl, x)`` → 追加 ``lsl_time + channels``。"""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        channel_labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.path = Path(path)
        self.labels: List[str] = list(channel_labels or DEVICE_CHANNEL_LABELS)
        self._lock = threading.Lock()
        self._file = None
        self._writer: Optional[csv.writer] = None
        self.rows_written = 0
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None
        self._started_at_local: Optional[float] = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["lsl_time"] + self.labels)
        self.rows_written = 0
        self.t_first = None
        self.t_last = None
        self._started_at_local = time.time()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                    self._file.close()
                except OSError:
                    pass
                self._file = None
                self._writer = None

    def on_chunk(self, t_lsl: np.ndarray, x: np.ndarray) -> None:
        if self._writer is None:
            return
        t = np.asarray(t_lsl, dtype=np.float64).reshape(-1)
        xx = np.asarray(x, dtype=np.float64)
        if xx.ndim == 1:
            xx = xx.reshape(1, -1)
        n = min(len(t), xx.shape[0])
        with self._lock:
            for i in range(n):
                ti = float(t[i])
                row = [f"{ti:.6f}"] + [f"{float(v):.6f}" for v in xx[i, : len(self.labels)]]
                self._writer.writerow(row)
                self.rows_written += 1
                if self.t_first is None:
                    self.t_first = ti
                self.t_last = ti
            if self._file is not None and self.rows_written % 250 == 0:
                self._file.flush()

    def write_meta(
        self,
        *,
        sample_rate_hz: float = 250.0,
        use_synthetic: bool = False,
        serial_port: str = "",
    ) -> Dict[str, Any]:
        """写 ``eeg.meta.json``（与 alignment / finalize 约定一致）。"""
        span = None
        if self.t_first is not None and self.t_last is not None:
            span = float(self.t_last - self.t_first)
        n = int(self.rows_written)
        # 粗质量：有样本且时间跨度与行数大致匹配则 timeline_ok
        expected = (span * float(sample_rate_hz)) if span is not None else None
        drop_rate = 0.0
        timeline_ok = True
        if expected is not None and expected > 1:
            drop_rate = max(0.0, 1.0 - (n / expected)) * 100.0
            timeline_ok = drop_rate < 15.0
        quality = {
            "drop_rate_pct": round(drop_rate, 3),
            "timeline": "ok" if timeline_ok else "suspect",
            "lsl_timeline_ok": timeline_ok,
            "source": "eeg_bus_csv",
        }
        meta: Dict[str, Any] = {
            "sample_rate_hz": float(sample_rate_hz),
            "channel_count": len(self.labels),
            "channel_labels": list(self.labels),
            "unit": "uV",
            "samples_written": n,
            "lsl_span_sec": span,
            "use_synthetic": bool(use_synthetic),
            "serial_port": str(serial_port or ""),
            "csv_file": str(self.path),
            "started_at_local": self._started_at_local,
            "stopped_at_local": time.time(),
            "source": "CsvRecorderSubscriber",
            "quality": quality,
        }
        dest = self.path.with_name("eeg.meta.json")
        if self.path.name != "eeg.csv":
            # eeg.csv → eeg.meta.json；其它名用 .meta.json 后缀
            dest = self.path.with_suffix(".meta.json")
            if self.path.suffix == ".csv":
                dest = self.path.parent / "eeg.meta.json"
        atomic_write_json(dest, meta)
        # 兼容旧路径 eeg.csv.meta.json
        legacy = self.path.parent / "eeg.csv.meta.json"
        try:
            atomic_write_json(legacy, meta)
        except OSError:
            pass
        return meta

    def __enter__(self) -> "CsvRecorderSubscriber":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
