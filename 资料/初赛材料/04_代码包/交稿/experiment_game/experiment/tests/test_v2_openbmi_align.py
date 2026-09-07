"""v2 OpenBMI-Align 配置与 v3 一致。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_game.experiment.v2_config import V2Config  # noqa: E402


def test_v2_config_openbmi_grid():
    cfg = V2Config.load_yaml()
    assert cfg.cue_s == 1.0
    assert cfg.imagine_s == 4.0
    assert cfg.online_window_mode == "openbmi_hop100"
    assert len(cfg.judgment_times) == 11
    assert cfg.judgment_times[0] == 3.0
    assert cfg.inter_trial_rest_s == 4.0
    assert cfg.primary_judge_mode == "majority"


def test_v3_baseline_rest_default():
    from experiment_game.experiment.v3_config import V3Config

    cfg = V3Config.load_yaml()
    assert cfg.baseline_rest_s == 30.0
