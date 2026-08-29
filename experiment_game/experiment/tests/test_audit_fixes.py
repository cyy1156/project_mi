"""审查报告 P0/P1 回归测试。"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiment_game.experiment.alignment import verify_alignment
from experiment_game.experiment.sim.bci2a_replay_source import Bci2aReplaySource
from experiment_game.experiment.sim.run_to_session_map import SimTrial, SimTrialScript
from experiment_game.experiment.inference_v2 import RingBuffer
from experiment_game.offline.epochs import to_model_tensor
from experiment_game.core.channel_layout import MODEL_INPUT_CHANNEL_ORDER, DEVICE_CHANNEL_LABELS


class TestAuditFixes(unittest.TestCase):
    def test_bci2a_csv_writer_initialized(self) -> None:
        script = SimTrialScript(
            subject_id="A01",
            run_id="run1",
            mat_path="",
            fs=250,
            x8=np.zeros((100, 8), dtype=np.float64),
            trials=[
                SimTrial(
                    cue_sample=50,
                    label=1,
                    mat_trial_index=0,
                    rest_start_sample=0,
                    rest_end_sample=40,
                )
            ],
            trials_unused=[],
            labels_by_block=[[1]],
            blocks=1,
            trials_per_block=1,
            session_trials_total=1,
        )
        buf = RingBuffer()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "eeg.csv"
            src = Bci2aReplaySource(script, buf, eeg_csv_path=csv_path, speed=1000.0)
            src.start()
            import time

            time.sleep(0.05)
            src.stop()
            self.assertTrue(csv_path.is_file())
            with csv_path.open(encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            self.assertEqual(header[0], "lsl_time")
            self.assertEqual(header[1:], list(DEVICE_CHANNEL_LABELS))

    def test_verify_alignment_respects_timing(self) -> None:
        rows = [
            {
                "phase": "acquire",
                "trial_id": 1,
                "mi_dur": 6.0,
                "rest_dur": 5.0,
                "rejected": 0,
            }
        ]
        ok = verify_alignment([], rows, None, require_acq=False, mi_s=6.0, rest_s=5.0)
        names = {c["name"]: c["ok"] for c in ok["checks"]}
        self.assertTrue(names["acquire_mi_duration"])
        self.assertTrue(names["acquire_rest_duration"])

    def test_to_model_tensor_permutes_channels(self) -> None:
        trials = [np.arange(8 * 10, dtype=np.float64).reshape(10, 8)]
        x = to_model_tensor(trials)
        self.assertEqual(x.shape, (1, 1, 8, 10))
        # 2026-08-29 起设备序=模型序（恒等映射）：设备第 i 列 = 模型第 i 轴
        for i in range(8):
            self.assertAlmostEqual(float(trials[0][0, i]), float(x[0, 0, i, 0]))
        self.assertEqual(MODEL_INPUT_CHANNEL_ORDER, DEVICE_CHANNEL_LABELS)

    def test_phase4_cal_rows_acquire_only(self) -> None:
        import experiment_game.offline.phase4_v2 as p4

        source = Path(p4.__file__).read_text(encoding="utf-8")
        self.assertIn('str(r.get("phase") or "") == "acquire"', source)
        self.assertIn("task_phases={\"acquire\"}", source)

    def test_leave_next_resolves_heldout_from_campaign(self) -> None:
        from experiment_game.experiment.sim.ramp import completed_by_run, leave_next_train_runs

        manifest = {
            "session_queue": ["run1", "run2", "run3"],
            "sessions_completed": [
                {"run_id": "run1", "session_dir": "/tmp/s1"},
                {"run_id": "run2", "session_dir": "/tmp/s2"},
                {"run_id": "run3", "session_dir": "/tmp/s3"},
            ],
        }
        train = leave_next_train_runs(manifest, "run3")
        self.assertEqual([r for r, _ in train], ["run1", "run2"])
        done = completed_by_run(manifest)
        self.assertEqual(done["run3"], "/tmp/s3")

    def test_atomic_copy_files_into(self) -> None:
        from experiment_game.core.atomic_io import atomic_copy_files_into

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            src.mkdir()
            (src / "best_task.pt").write_bytes(b"task")
            (src / "best_three.pt").write_bytes(b"three")
            (src / "meta.json").write_text("{}", encoding="utf-8")
            copied = atomic_copy_files_into(
                src, dst, ("best_task.pt", "best_three.pt", "meta.json", "missing.pt")
            )
            self.assertEqual(set(copied), {"best_task.pt", "best_three.pt", "meta.json"})
            self.assertEqual((dst / "best_task.pt").read_bytes(), b"task")
            self.assertFalse((dst / "missing.pt").exists())

    def test_infer_stale_check_can_disable(self) -> None:
        import time

        from experiment_game.experiment.inference_v2 import InferenceService

        buf = RingBuffer()
        buf._watch_mono = time.monotonic() - 10.0
        infer = object.__new__(InferenceService)
        infer.buffer = buf
        infer.window_mode = "legacy"
        infer.stale_check_enabled = True
        stale = InferenceService.judge(infer, 0.0, 0.0)
        self.assertTrue(stale and stale.get("eeg_stale"))

        infer.stale_check_enabled = False
        with mock.patch.object(InferenceService, "_judge_legacy", return_value={"pred": 1}):
            out = InferenceService.judge(infer, 0.0, 0.0)
        self.assertEqual(out, {"pred": 1})


if __name__ == "__main__":
    unittest.main()
