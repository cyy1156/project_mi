"""alignment.verify_alignment 录制质量校验。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment_game.experiment.alignment import verify_alignment, write_alignment_bundle


class TestAlignmentQuality(unittest.TestCase):
    def _minimal_eeg(self, root: Path) -> Path:
        path = root / "eeg.csv"
        path.write_text("lsl_time,C3\n1.0,0.1\n2.0,0.2\n", encoding="utf-8")
        return path

    def test_drop_rate_fail_lsl_not_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = self._minimal_eeg(root)
            report = verify_alignment(
                [],
                [],
                eeg,
                require_acq=True,
                eeg_quality={
                    "drop_rate_pct": 2.5,
                    "lsl_timeline_ok": False,
                    "severity": "warn",
                },
            )
            self.assertFalse(report["passed"])
            checks = {c["name"]: c["ok"] for c in report["checks"]}
            self.assertFalse(checks["eeg_drop_rate_ok"])

    def test_drop_rate_pass_when_lsl_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = self._minimal_eeg(root)
            report = verify_alignment(
                [],
                [],
                eeg,
                require_acq=True,
                eeg_quality={
                    "drop_rate_pct": 7.4,
                    "lsl_timeline_ok": True,
                    "severity": "ok",
                },
            )
            checks = {c["name"]: c["ok"] for c in report["checks"]}
            self.assertTrue(checks["eeg_drop_rate_ok"])

    def test_drop_rate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = self._minimal_eeg(root)
            report = verify_alignment(
                [],
                [],
                eeg,
                require_acq=True,
                eeg_quality={
                    "drop_rate_pct": 0.1,
                    "lsl_timeline_ok": True,
                },
            )
            checks = {c["name"]: c["ok"] for c in report["checks"]}
            self.assertTrue(checks["eeg_drop_rate_ok"])
            self.assertTrue(checks["eeg_lsl_timeline_ok"])

    def test_write_bundle_reads_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events.jsonl").write_text("", encoding="utf-8")
            (root / "eeg.csv").write_text("lsl_time,C3\n1.0,0.1\n", encoding="utf-8")
            meta = {
                "quality": {
                    "drop_rate_pct": 0.05,
                    "lsl_timeline_ok": True,
                    "severity": "ok",
                }
            }
            (root / "eeg.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False),
                encoding="utf-8",
            )
            report = write_alignment_bundle(root, acq_enabled=True)
            self.assertIn("quality", report)
            names = {c["name"] for c in report["checks"]}
            self.assertIn("eeg_drop_rate_ok", names)


if __name__ == "__main__":
    unittest.main()
