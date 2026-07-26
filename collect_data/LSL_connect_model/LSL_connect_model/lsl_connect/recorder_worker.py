"""
选修 B / FR-31：订阅 LSL EEG 流，追加写入本地 CSV（滤波后 µV）。
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from lsl_connect.lsl_streams import DEFAULT_EEG_LABELS, EEG_STREAM_NAME


@dataclass
class RecorderWorkerConfig:
    stream_name: str = EEG_STREAM_NAME
    resolve_timeout: float = 5.0
    pull_timeout: float = 0.05
    flush_interval_sec: float = 2.0
    max_samples_per_pull: int = 250
    lsl_buffer_sec: int = 300
    sample_rate_hz: int = 250


class RecorderWorker:
    """在独立线程中从 LSL 拉 chunk 并写入 CSV。"""

    def __init__(
        self,
        config: Optional[RecorderWorkerConfig] = None,
        on_error: Optional[Callable[[str], None]] = None,
        get_samples_pushed: Optional[Callable[[], int]] = None,
    ) -> None:
        self._config = config or RecorderWorkerConfig()
        self._on_error = on_error
        self._get_samples_pushed = get_samples_pushed

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inlet: Optional[StreamInlet] = None

        self._file: Optional[Any] = None
        self._writer: Optional[csv.writer] = None
        self._path: Optional[Path] = None
        self._meta: Dict[str, Any] = {}
        self._channel_labels: List[str] = []

        self._lock = threading.Lock()
        self._samples_written = 0
        self._empty_pulls = 0
        self._estimated_gap_samples = 0
        self._non_monotonic_fixes = 0
        self._last_lsl_ts: Optional[float] = None
        self._first_lsl_ts: Optional[float] = None
        self._started_at: Optional[float] = None
        self._first_write_at: Optional[float] = None
        self._acq_baseline_at_first_write: Optional[int] = None
        self._last_flush = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            path = str(self._path) if self._path else None
            return {
                "active": self.is_running,
                "path": path,
                "samples_written": self._samples_written,
                "estimated_gap_samples": self._estimated_gap_samples,
                "non_monotonic_fixes": self._non_monotonic_fixes,
                "empty_pulls": self._empty_pulls,
            }

    def start(
        self,
        csv_path: Path,
        channel_labels: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.is_running:
            raise RuntimeError("录制线程已在运行")

        labels = list(channel_labels or DEFAULT_EEG_LABELS)
        self._channel_labels = labels
        self._meta = dict(meta or {})
        self._meta.setdefault("channel_labels", labels)
        self._meta.setdefault("unit", "uV")
        self._meta.setdefault("filtered", True)

        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._path = csv_path
        self._file = csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["lsl_time", *labels])

        with self._lock:
            self._samples_written = 0
            self._empty_pulls = 0
            self._estimated_gap_samples = 0
            self._non_monotonic_fixes = 0
        self._last_lsl_ts = None
        self._first_lsl_ts = None
        self._first_write_at = None
        self._acq_baseline_at_first_write = None
        self._started_at = time.time()
        self._last_flush = self._started_at

        streams = resolve_byprop(
            "name",
            self._config.stream_name,
            minimum=1,
            timeout=self._config.resolve_timeout,
        )
        if not streams:
            self._close_file()
            raise RuntimeError(
                f"未找到 LSL 流 {self._config.stream_name!r}，请先 start 采集"
            )

        buf_sec = max(30, int(self._config.lsl_buffer_sec))
        max_chunk = max(1, int(self._config.max_samples_per_pull))
        self._inlet = StreamInlet(
            streams[0],
            max_buflen=buf_sec,
            max_chunklen=max_chunk,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="RecorderWorker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 8.0) -> Dict[str, Any]:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

        self._drain_inlet()

        inlet = self._inlet
        if inlet is not None:
            try:
                inlet.close_stream()
            except Exception:
                pass
        self._inlet = None

        with self._lock:
            samples = self._samples_written
            path = self._path
            gaps = self._estimated_gap_samples
            empty_pulls = self._empty_pulls
            fixes = self._non_monotonic_fixes

        first_ts = self._first_lsl_ts
        last_ts = self._last_lsl_ts
        lsl_span = (
            float(last_ts - first_ts)
            if first_ts is not None and last_ts is not None
            else 0.0
        )

        self._write_meta_sidecar(samples, lsl_span=lsl_span, fixes=fixes)
        self._close_file()

        return {
            "path": str(path) if path else None,
            "samples_written": samples,
            "estimated_gap_samples": gaps,
            "non_monotonic_fixes": fixes,
            "empty_pulls": empty_pulls,
            "first_lsl_ts": first_ts,
            "last_lsl_ts": last_ts,
            "lsl_span_sec": lsl_span,
            "first_write_at": self._first_write_at,
            "acq_baseline_at_first_write": self._acq_baseline_at_first_write,
        }

    def _write_meta_sidecar(
        self,
        samples: int,
        *,
        lsl_span: float,
        fixes: int,
    ) -> None:
        if self._path is None:
            return
        meta_path = self._path.with_suffix(".meta.json")
        payload = dict(self._meta)
        payload["samples_written"] = samples
        payload["estimated_gap_samples"] = self._estimated_gap_samples
        payload["non_monotonic_fixes"] = fixes
        payload["empty_pulls"] = self._empty_pulls
        payload["lsl_span_sec"] = round(lsl_span, 4)
        payload["lsl_buffer_sec"] = self._config.lsl_buffer_sec
        payload["stopped_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self._started_at is not None:
            payload["started_at_local"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self._started_at)
            )
        try:
            meta_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self._emit_error(f"写入元数据失败: {exc}")

    def _close_file(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except OSError:
                pass
        self._file = None
        self._writer = None

    def _emit_error(self, msg: str) -> None:
        if self._on_error is not None:
            self._on_error(msg)

    def _normalize_timestamps(
        self,
        timestamps: List[float],
        n_rows: int,
        expected_dt: float,
    ) -> List[float]:
        ts_list = list(timestamps[:n_rows])
        if not ts_list:
            base = self._last_lsl_ts if self._last_lsl_ts is not None else 0.0
            return [base + (i + 1) * expected_dt for i in range(n_rows)]

        while len(ts_list) < n_rows:
            ts_list.append(ts_list[-1] + expected_dt)
        return ts_list

    def _next_monotonic_ts(self, ts: float, expected_dt: float) -> float:
        if self._last_lsl_ts is not None and ts <= self._last_lsl_ts:
            with self._lock:
                self._non_monotonic_fixes += 1
            return self._last_lsl_ts + expected_dt
        return ts

    def _mark_first_write(self) -> None:
        if self._first_write_at is not None:
            return
        self._first_write_at = time.time()
        if self._get_samples_pushed is not None:
            try:
                self._acq_baseline_at_first_write = int(self._get_samples_pushed())
            except Exception:
                pass

    def _write_rows(
        self,
        samples: np.ndarray,
        ts_list: List[float],
        *,
        expected_dt: float,
        gap_threshold: float,
        n_labels: int,
    ) -> int:
        assert self._writer is not None
        n_rows = samples.shape[0]
        rows: List[List[Any]] = []
        for i in range(n_rows):
            ts = self._next_monotonic_ts(ts_list[i], expected_dt)
            if self._last_lsl_ts is not None:
                self._record_gap(ts - self._last_lsl_ts, expected_dt, gap_threshold)
            if self._first_lsl_ts is None:
                self._first_lsl_ts = ts
                self._mark_first_write()
            self._last_lsl_ts = ts

            row = samples[i, :n_labels].tolist()
            if len(row) < n_labels:
                row.extend([0.0] * (n_labels - len(row)))
            rows.append([ts, *row])

        self._writer.writerows(rows)
        return n_rows

    def _record_gap(self, gap: float, expected_dt: float, gap_threshold: float) -> None:
        if gap > gap_threshold:
            missed = int(gap / expected_dt) - 1
            if missed > 0:
                with self._lock:
                    self._estimated_gap_samples += missed

    def _run_loop(self) -> None:
        assert self._inlet is not None
        assert self._writer is not None

        cfg = self._config
        n_labels = len(self._channel_labels)
        fs = max(1, int(cfg.sample_rate_hz))
        expected_dt = 1.0 / float(fs)
        gap_threshold = expected_dt * 2.5

        while not self._stop_event.is_set():
            try:
                chunk, timestamps = self._inlet.pull_chunk(
                    timeout=cfg.pull_timeout,
                    max_samples=cfg.max_samples_per_pull,
                )
            except Exception as exc:
                self._emit_error(f"pull_chunk 异常: {exc}")
                break

            if self._stop_event.is_set():
                break
            if not chunk:
                with self._lock:
                    self._empty_pulls += 1
                continue

            samples = np.asarray(chunk, dtype=np.float64)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)

            n_rows = samples.shape[0]
            ts_list = self._normalize_timestamps(
                [float(t) for t in timestamps] if timestamps else [],
                n_rows,
                expected_dt,
            )

            try:
                n_rows = self._write_rows(
                    samples,
                    ts_list,
                    expected_dt=expected_dt,
                    gap_threshold=gap_threshold,
                    n_labels=n_labels,
                )
            except OSError as exc:
                self._emit_error(f"写 CSV 失败: {exc}")
                break

            with self._lock:
                self._samples_written += n_rows

            now = time.time()
            if now - self._last_flush >= cfg.flush_interval_sec:
                try:
                    if self._file is not None:
                        self._file.flush()
                except OSError as exc:
                    self._emit_error(f"flush 失败: {exc}")
                self._last_flush = now

        if self._file is not None:
            try:
                self._file.flush()
            except OSError:
                pass

    def _drain_inlet(self) -> None:
        """停录前排空 Inlet 缓冲，避免 LSL 队列中残留样本未写入。"""
        inlet = self._inlet
        writer = self._writer
        if inlet is None or writer is None:
            return

        cfg = self._config
        n_labels = len(self._channel_labels)
        fs = max(1, int(cfg.sample_rate_hz))
        expected_dt = 1.0 / float(fs)
        gap_threshold = expected_dt * 2.5

        for _ in range(512):
            try:
                chunk, timestamps = inlet.pull_chunk(
                    timeout=0.0,
                    max_samples=cfg.max_samples_per_pull,
                )
            except Exception:
                break
            if not chunk:
                break

            samples = np.asarray(chunk, dtype=np.float64)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            n_rows = samples.shape[0]
            ts_list = self._normalize_timestamps(
                [float(t) for t in timestamps] if timestamps else [],
                n_rows,
                expected_dt,
            )
            try:
                n_rows = self._write_rows(
                    samples,
                    ts_list,
                    expected_dt=expected_dt,
                    gap_threshold=gap_threshold,
                    n_labels=n_labels,
                )
            except OSError:
                break
            with self._lock:
                self._samples_written += n_rows
