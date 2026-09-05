"""阶段 2：崩溃落盘兜底 + events.jsonl 容错。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment_game.core.atomic_io import atomic_write_json
from experiment_game.core.jsonl import read_jsonl_tolerant
from experiment_game.experiment.session_finalize import (
    ensure_crash_artifacts,
    ensure_eeg_meta,
)
from experiment_game.experiment.session_layout import finalize_session_layout


class TestJsonlTolerant(unittest.TestCase):
    def test_skips_bad_lines_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "events.jsonl"
            p.write_text(
                '{"event":"ok","t_lsl":1.0}\n'
                "not-json\n"
                '{"event":"ok2","t_lsl":2.0}\n'
                "[1,2,3]\n",
                encoding="utf-8",
            )
            rows, n_bad = read_jsonl_tolerant(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual(n_bad, 2)
            self.assertEqual(rows[0]["event"], "ok")
            self.assertEqual(rows[1]["event"], "ok2")


class TestSessionFinalize(unittest.TestCase):
    def test_crash_artifacts_write_aborted_meta_manifest_eeg(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(
                root / "session.meta.json",
                {"subject_id": "t", "session_id": "ws01", "phase_mode": "v3_session"},
            )
            (root / "events.jsonl").write_text(
                '{"event":"session_start","t_lsl":0.0}\nbad-line\n'
                '{"event":"session_end","t_lsl":1.0}\n',
                encoding="utf-8",
            )
            (root / "eeg.csv").write_text(
                "lsl_time,FC3,C3,CP3,CZ,CPZ,FC4,C4,CP4\n"
                "0.0,1,1,1,1,1,1,1,1\n",
                encoding="utf-8",
            )
            atomic_write_json(root / "eeg.csv.meta.json", {"fs": 250, "n_rows": 1})

            rep = ensure_crash_artifacts(
                root,
                aborted=True,
                reason="unit_test_abort",
                acq_enabled=True,
                save_layout="phase_folders",
            )
            self.assertTrue(rep.get("ok"), rep)
            meta = json.loads((root / "session.meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta.get("aborted"))
            self.assertEqual(meta.get("abort_reason"), "unit_test_abort")
            self.assertTrue(meta.get("incomplete"))
            self.assertTrue((root / "manifest.json").is_file())
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest.get("events_parse_warnings"), 1)
            self.assertTrue((root / "eeg.meta.json").is_file())

    def test_ensure_eeg_meta_stub_when_no_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "eeg.csv").write_text("lsl_time,FC3\n0,1\n", encoding="utf-8")
            out = ensure_eeg_meta(root)
            self.assertIsNotNone(out)
            blob = json.loads((root / "eeg.meta.json").read_text(encoding="utf-8"))
            self.assertTrue(blob.get("incomplete"))

    def test_finalize_layout_records_parse_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "events.jsonl").write_text(
                '{"event":"session_start","t_lsl":0}\n{{{{{\n',
                encoding="utf-8",
            )
            man = finalize_session_layout(
                root, save_layout="flat", save_continuous=True, acq_enabled=False
            )
            self.assertGreaterEqual(int(man.get("events_parse_warnings") or 0), 1)


if __name__ == "__main__":
    unittest.main()
