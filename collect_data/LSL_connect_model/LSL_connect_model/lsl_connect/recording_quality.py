"""
录制质量评估：对比采集推送量与 CSV 写入量，估算丢包率。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RecordingStopReport:
    """停录后的质量报告，供 UI 弹窗与 meta.json。"""

    samples_written: int = 0
    samples_pushed_during: int = 0
    estimated_gap_samples: int = 0
    non_monotonic_fixes: int = 0
    missing_vs_lsl: int = 0
    drop_rate_pct: float = 0.0
    duration_sec: float = 0.0
    expected_by_duration: int = 0
    lsl_span_sec: float = 0.0
    expected_by_lsl_span: int = 0
    alignment_gap_samples: int = 0
    lsl_timeline_ok: bool = False
    sample_rate_hz: int = 250
    path: Optional[str] = None
    severity: str = "ok"  # ok | warn | bad

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples_written": self.samples_written,
            "samples_pushed_during_recording": self.samples_pushed_during,
            "estimated_gap_samples": self.estimated_gap_samples,
            "non_monotonic_fixes": self.non_monotonic_fixes,
            "missing_vs_lsl": self.missing_vs_lsl,
            "drop_rate_pct": round(self.drop_rate_pct, 3),
            "duration_sec": round(self.duration_sec, 2),
            "expected_by_duration": self.expected_by_duration,
            "lsl_span_sec": round(self.lsl_span_sec, 3),
            "expected_by_lsl_span": self.expected_by_lsl_span,
            "alignment_gap_samples": self.alignment_gap_samples,
            "lsl_timeline_ok": self.lsl_timeline_ok,
            "sample_rate_hz": self.sample_rate_hz,
            "severity": self.severity,
        }

    def summary_message(self) -> str:
        lines = [
            f"CSV 已写入: {self.samples_written} 行",
        ]
        if self.lsl_timeline_ok:
            lines.append(
                f"LSL 时间轴: {self.lsl_span_sec:.1f} s，"
                f"约需 {self.expected_by_lsl_span} 行 — 与 CSV 一致 ✓"
            )
            if self.alignment_gap_samples > 0:
                gap_sec = self.alignment_gap_samples / float(max(1, self.sample_rate_hz))
                lines.append(
                    f"启动/墙钟统计对齐差: {self.alignment_gap_samples} 样本"
                    f"（约 {gap_sec:.1f} s，非 CSV 丢包）"
                )
            lines.append(
                f"采集推送(录制时段): {self.samples_pushed_during} 样本"
                f"（含连接等待期，仅供参考）"
            )
        else:
            lines.append(f"采集推送(录制时段): {self.samples_pushed_during} 样本")
            lines.append(f"相对 LSL 缺口: {self.missing_vs_lsl} 样本")
            lines.append(f"Recorder 写不及率: {self.drop_rate_pct:.2f}%")

        lines.extend(
            [
                f"时间戳缺口估计: {self.estimated_gap_samples} 样本",
                f"时间戳回退修正: {self.non_monotonic_fixes} 次",
                f"录制墙钟: {self.duration_sec:.1f} s（约需 {self.expected_by_duration} 行，含等待）",
            ]
        )
        if not self.lsl_timeline_ok:
            lines.append(
                f"LSL 时间轴跨度: {self.lsl_span_sec:.1f} s"
                f"（约需 {self.expected_by_lsl_span} 行）"
            )
        if self.path:
            lines.append(f"文件: {self.path}")
        return "\n".join(lines)

    def popup_title(self) -> str:
        if self.severity == "bad":
            return "录制停录 — 质量异常"
        if self.severity == "warn":
            return "录制停录 — 有少量缺口"
        if self.lsl_timeline_ok and self.alignment_gap_samples > 0:
            return "录制停录 — 质量正常"
        return "录制停录 — 质量正常"


def _is_lsl_timeline_ok(
    written: int,
    expected_lsl: int,
    gaps: int,
    fixes: int,
    fs: int,
) -> bool:
    if expected_lsl <= 0:
        return False
    ratio = written / float(expected_lsl)
    return ratio >= 0.95 and ratio <= 1.05 and gaps <= fs * 2 and fixes <= fs


def compute_recording_quality(
    *,
    samples_written: int,
    samples_pushed_baseline: int,
    samples_pushed_now: int,
    estimated_gap_samples: int,
    sample_rate_hz: int,
    started_at: Optional[float],
    stopped_at: Optional[float] = None,
    csv_path: Optional[str] = None,
    lsl_span_sec: Optional[float] = None,
    non_monotonic_fixes: int = 0,
    warn_drop_pct: float = 1.0,
    bad_drop_pct: float = 5.0,
) -> RecordingStopReport:
    """
    对比「录制时段 Outlet 推送量」「CSV 行数」「墙钟时长」「LSL 时间轴跨度」。
    LSL 时间轴与 CSV 一致时，不再因写不及率单独判 warn。
    """
    stopped_at = stopped_at or time.time()
    pushed_during = max(0, int(samples_pushed_now) - int(samples_pushed_baseline))
    written = max(0, int(samples_written))
    gaps = max(0, int(estimated_gap_samples))
    fixes = max(0, int(non_monotonic_fixes))

    missing_lsl = max(0, pushed_during - written)
    drop_rate = (100.0 * missing_lsl / pushed_during) if pushed_during > 0 else 0.0

    duration = 0.0
    if started_at is not None:
        duration = max(0.0, stopped_at - started_at)
    fs = max(1, int(sample_rate_hz))
    expected_wall = int(duration * fs)

    span = max(0.0, float(lsl_span_sec or 0.0))
    expected_lsl = int(span * fs) if span > 0 else 0

    lsl_ok = _is_lsl_timeline_ok(written, expected_lsl, gaps, fixes, fs)
    wall_vs_lsl = max(0, expected_wall - expected_lsl)
    alignment_gap = max(missing_lsl, wall_vs_lsl) if lsl_ok else 0

    severity = "ok"

    if lsl_ok:
        severity = "ok"
    elif expected_lsl > 0:
        lsl_ratio = written / float(expected_lsl)
        if lsl_ratio < 0.90:
            severity = "bad"
        elif lsl_ratio < 0.98 or gaps > fs * 2 or drop_rate >= bad_drop_pct:
            severity = "warn"
    elif expected_wall > 0:
        wall_ratio = written / float(expected_wall)
        if wall_ratio < 0.90 or drop_rate >= bad_drop_pct:
            severity = "bad"
        elif wall_ratio < 0.98 or drop_rate >= warn_drop_pct or gaps > fs * 2:
            severity = "warn"
    elif drop_rate >= bad_drop_pct:
        severity = "bad"
    elif drop_rate >= warn_drop_pct:
        severity = "warn"

    if expected_wall > 0 and written > expected_wall * 1.10:
        severity = "bad"
    if fixes > fs and not lsl_ok:
        severity = "warn" if severity == "ok" else severity

    return RecordingStopReport(
        samples_written=written,
        samples_pushed_during=pushed_during,
        estimated_gap_samples=gaps,
        non_monotonic_fixes=fixes,
        missing_vs_lsl=missing_lsl,
        drop_rate_pct=drop_rate,
        duration_sec=duration,
        expected_by_duration=expected_wall,
        lsl_span_sec=span,
        expected_by_lsl_span=expected_lsl,
        alignment_gap_samples=alignment_gap,
        lsl_timeline_ok=lsl_ok,
        sample_rate_hz=fs,
        path=str(csv_path) if csv_path else None,
        severity=severity,
    )


def patch_meta_quality(csv_path: str | Path, quality: Dict[str, Any]) -> None:
    """在已有 .meta.json 中追加 quality 字段。"""
    meta_path = Path(csv_path).with_suffix(".meta.json")
    if not meta_path.is_file():
        return
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        payload["quality"] = quality
        meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError):
        pass
