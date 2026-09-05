"""BoardToLslTimestampMapper 单调性与 CSV 质量逻辑单元测试。"""

from __future__ import annotations

import numpy as np

from lsl_connect.lsl_streams import BoardToLslTimestampMapper
from lsl_connect.recording_quality import compute_recording_quality


def test_timestamp_mapper_monotonic_degenerate_board_ts() -> None:
    mapper = BoardToLslTimestampMapper()
    fs = 250
    dt = 1.0 / fs
    all_ts: list[float] = []

    for _ in range(20):
        board_ts = np.full(25, 1.0)
        batch = mapper.to_lsl_uniform(board_ts, fs)
        assert len(batch) == 25
        assert all(batch[i] < batch[i + 1] for i in range(len(batch) - 1))
        if all_ts:
            assert batch[0] > all_ts[-1]
        all_ts.extend(batch)

    span = all_ts[-1] - all_ts[0]
    expected = (len(all_ts) - 1) * dt
    assert abs(span - expected) < dt * 2


def test_recording_quality_uses_lsl_span() -> None:
    report = compute_recording_quality(
        samples_written=1200,
        samples_pushed_baseline=0,
        samples_pushed_now=1210,
        estimated_gap_samples=0,
        sample_rate_hz=250,
        started_at=1000.0,
        stopped_at=1005.0,
        lsl_span_sec=4.8,
        non_monotonic_fixes=0,
    )
    assert report.expected_by_lsl_span == 1200
    assert report.severity == "ok"


def test_recording_quality_flags_inflation() -> None:
    report = compute_recording_quality(
        samples_written=10000,
        samples_pushed_baseline=0,
        samples_pushed_now=10000,
        estimated_gap_samples=0,
        sample_rate_hz=250,
        started_at=0.0,
        stopped_at=5.0,
        lsl_span_sec=5.0,
    )
    assert report.severity == "bad"


def test_recording_quality_alignment_gap_not_warn() -> None:
    """用户场景：LSL 完整但墙钟统计有差 → ok + 对齐差说明。"""
    report = compute_recording_quality(
        samples_written=2808,
        samples_pushed_baseline=0,
        samples_pushed_now=3043,
        estimated_gap_samples=0,
        sample_rate_hz=250,
        started_at=0.0,
        stopped_at=12.31,
        lsl_span_sec=11.228,
        non_monotonic_fixes=0,
    )
    assert report.severity == "ok"
    assert report.lsl_timeline_ok is True
    assert report.alignment_gap_samples >= 235
    assert "非 CSV 丢包" in report.summary_message()
    assert report.popup_title() == "录制停录 — 质量正常"


if __name__ == "__main__":
    test_timestamp_mapper_monotonic_degenerate_board_ts()
    test_recording_quality_uses_lsl_span()
    test_recording_quality_flags_inflation()
    test_recording_quality_alignment_gap_not_warn()
    print("ok")
