"""AcquisitionFacade.health_check 合成板冒烟（短等待，对齐 Bus CSV 路径）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment_game.acquisition import AcquisitionFacade


class TestAcquisitionHealth(unittest.TestCase):
    def test_health_check_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "eeg.csv"
            acq = AcquisitionFacade(use_synthetic=True)
            acq.create()
            # 与现网一致：板卡只推 LSL，CSV 由 LiveEegCapture 写
            acq.start(csv_path, record_csv=False)
            try:
                hc = acq.health_check(
                    wait_s=0.35,
                    min_samples=40,
                    warmup_s=0.15,
                    retries=1,
                )
                self.assertGreaterEqual(hc["delta_samples"], 40)
                self.assertTrue(hc["lsl_ok"])
            finally:
                acq.stop()
                acq.shutdown()


if __name__ == "__main__":
    unittest.main()
