"""EEGBus 订阅者：将样本原子追写成 eeg.csv（总册 W2 落盘订户）。

仿真回放等「经 Bus 扇出」路径使用；真机 lsl_connect Recorder 仍独立落盘，
直至采集侧改为 Bus 单写。
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

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

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["lsl_time"] + self.labels)
        self.rows_written = 0

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
                row = [f"{float(t[i]):.6f}"] + [f"{float(v):.6f}" for v in xx[i, : len(self.labels)]]
                self._writer.writerow(row)
                self.rows_written += 1
            if self._file is not None and self.rows_written % 250 == 0:
                self._file.flush()

    def __enter__(self) -> "CsvRecorderSubscriber":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
