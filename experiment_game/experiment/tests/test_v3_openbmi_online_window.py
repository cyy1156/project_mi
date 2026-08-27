"""v3 OpenBMI 在线取窗对齐测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_game.experiment.inference_v2 import (  # noqa: E402
    FS,
    N_TIMES_3S,
    OnlinePreprocessor,
    RingBuffer,
)
from experiment_game.experiment.openbmi_align_config import (  # noqa: E402
    build_openbmi_judgment_times,
    rebuild_judgment_times,
)
from experiment_game.experiment.v3_config import V3Config  # noqa: E402


def test_build_openbmi_judgment_times_mi4():
    times = build_openbmi_judgment_times(4.0, win_s=3.0, hop_s=0.1)
    assert len(times) == 11
    assert times[0] == 3.0
    assert times[-1] == 4.0
    assert round(times[1] - times[0], 1) == 0.1


def test_v3_config_loads_openbmi_grid():
    cfg = V3Config.load_yaml()
    rebuild_judgment_times(cfg)
    assert cfg.online_window_mode == "openbmi_hop100"
    assert len(cfg.judgment_times) == 11
    assert cfg.judgment_times[0] == 3.0


def test_process_openbmi_task_window_shape():
    pre = OnlinePreprocessor()
    seg_len_s = 0.5 + 3.0
    n_seg = int(round(seg_len_s * FS))
    n_tail = int(round(12.0 * FS))
    rng = np.random.default_rng(0)
    tail = rng.normal(0, 10, (max(n_tail, n_seg), 8))
    w = pre.process_openbmi_task_window(
        tail,
        seg_len_s=seg_len_s,
        win_start_rel=0.0,
        win_end_rel=3.0,
        baseline_sec=0.5,
    )
    assert w is not None
    assert w.shape == (8, N_TIMES_3S)


def test_openbmi_window_indices_last_window():
    pre = OnlinePreprocessor()
    seg_len_s = 0.5 + 4.0
    n_tail = int(round(seg_len_s * FS))
    tail = np.random.default_rng(1).normal(0, 5, (n_tail, 8))
    w = pre.process_openbmi_task_window(
        tail,
        seg_len_s=seg_len_s,
        win_start_rel=1.0,
        win_end_rel=4.0,
        baseline_sec=0.5,
    )
    assert w is not None
    assert w.shape == (8, N_TIMES_3S)
