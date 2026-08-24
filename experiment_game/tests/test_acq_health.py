"""AcquisitionFacade.health_check 合成板冒烟。"""

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
            acq.start(csv_path)
            try:
                hc = acq.health_check(wait_s=1.0, min_samples=100)
                self.assertGreater(hc["delta_samples"], 100)
                self.assertTrue(hc["lsl_ok"])
            finally:
                acq.stop()
                acq.shutdown()


if __name__ == "__main__":
    unittest.main()
