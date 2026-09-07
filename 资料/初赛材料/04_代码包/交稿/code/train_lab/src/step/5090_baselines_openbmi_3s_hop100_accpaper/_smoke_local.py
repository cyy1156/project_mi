"""5090 · 方案 24 · 3s 包冒烟：import + t0 + prob 格式。"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from prob_dump import DUMP_COLUMNS  # noqa: E402
from shared_hparams import OUT_ROOT_TAG, SHARED, TRAIN_DEVICE_LABEL  # noqa: E402
from t0_sec import T0_MAX, T0_MIN, compute_window_t0_sec, t0_train_weight  # noqa: E402

def main() -> None:
    assert SHARED.batch_train == 256
    assert SHARED.batch_eval == 512
    assert SHARED.n_times_expected == 750
    assert SHARED.t0_weight_alpha == 0.0
    assert OUT_ROOT_TAG == "5090_alg_incr_3s_hop100_accpaper"
    assert "5090" in TRAIN_DEVICE_LABEL
    assert len(DUMP_COLUMNS) == 11

    tid = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    t0 = compute_window_t0_sec(tid)
    assert abs(float(t0[0]) - T0_MIN) < 1e-5
    assert abs(float(t0[2]) - T0_MAX) < 1e-5
    w = t0_train_weight(t0, alpha=0.6)
    assert abs(float(w[0]) - 1.0) < 1e-5
    assert float(w[2]) < float(w[0])

    import baseline_shallow  # noqa: F401
    import baseline_eegnet  # noqa: F401
    import baseline_conformer  # noqa: F401
    import dump_probs  # noqa: F401
    import replay_v_weighted_vote  # noqa: F401

    print("OUT_ROOT_TAG", OUT_ROOT_TAG)
    print("scheme24 3s smoke ALL OK")

if __name__ == "__main__":
    main()
