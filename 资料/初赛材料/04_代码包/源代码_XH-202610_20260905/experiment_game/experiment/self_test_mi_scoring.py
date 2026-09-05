"""MI 多数票计分自测：单元测试 + 离线回放冒烟。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_game.experiment import trial_v2 as tv2  # noqa: E402
from experiment_game.experiment.events_log import EventLogger  # noqa: E402
from experiment_game.experiment.markers import MarkerPublisher  # noqa: E402
from experiment_game.experiment.scoring_replay import replay_session  # noqa: E402
from experiment_game.experiment.trial_v2 import (  # noqa: E402
    TrialContextV2,
    TrialStateMachineV2,
    TrialTimingV2,
)
from experiment_game.experiment.v2_config import V2Config  # noqa: E402


def _noop_wait(t_end, **kwargs) -> None:
    return None


def test_config_majority_mode():
    cfg = V2Config.load_yaml()
    assert cfg.primary_judge_mode == "majority"
    assert cfg.imagine_s == 4.0
    assert len(cfg.judgment_times) == 11


def test_replay_smoke():
    tv2._wu = _noop_wait
    times = tuple(round(3.0 + i * 0.1, 1) for i in range(11))
    timing = TrialTimingV2(
        prep_s=0.001,
        cue_s=0.0,
        imagine_s=4.0,
        iti_s=0.001,
        inter_trial_rest_s=0.0,
        judgment_times=times,
    )
    with tempfile.TemporaryDirectory() as td:
        events_path = Path(td) / "events.jsonl"
        events = EventLogger(events_path)
        markers = MarkerPublisher(enabled=False)

        def judgment_fn(mi_t, t_rel, ctx):
            return {"pred": 1, "p_max": 0.9, "gated": False}

        sm = TrialStateMachineV2(events, markers, timing, judgment_fn=judgment_fn)
        ctx = TrialContextV2(trial_id=1, label=1, mode="calibration", round_no=1)
        summary = sm.run_trial(ctx)
        events.close()
        assert summary["score"] == 1.0
        rows = replay_session(events_path)
        assert len(rows) == 1
        assert rows[0]["score"] == 1.0


def main() -> int:
    test_config_majority_mode()
    test_replay_smoke()
    print("MI 多数票自测通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
