"""trial_v2 信号质量门控接线。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_game.experiment import trial_v2 as tv2  # noqa: E402
from experiment_game.experiment.events_log import EventLogger  # noqa: E402
from experiment_game.experiment.markers import MarkerPublisher  # noqa: E402
from experiment_game.experiment.trial_v2 import TrialContextV2, TrialStateMachineV2, TrialTimingV2  # noqa: E402
from adapt_engine.scoring_v21 import ScoringConfig, build_judgment_times  # noqa: E402


def _fast_timing() -> TrialTimingV2:
    sc = ScoringConfig()
    return TrialTimingV2(
        prep_s=0.001,
        cue_s=0.001,
        imagine_s=6.0,
        iti_s=0.001,
        judgment_times=build_judgment_times(0.6, 6.0),
        scoring=sc,
    )


def test_signal_bad_skips_scoring():
    tv2._wu = lambda t_end, **kw: None
    with tempfile.TemporaryDirectory() as td:
        events = EventLogger(Path(td) / "events.jsonl")
        markers = MarkerPublisher(enabled=False)

        def judgment_fn(mi_t, t_rel, ctx):
            return {"signal_bad": True, "reason": "flatline", "signal_metrics": {}}

        sm = TrialStateMachineV2(
            events,
            markers,
            _fast_timing(),
            judgment_fn=judgment_fn,
        )
        ctx = TrialContextV2(trial_id=1, label=0, mode="calibration", round_no=1)
        summary = sm.run_trial(ctx)
        events.close()

        assert summary is not None
        assert summary["valid"] is False
        assert summary["invalid_reason"] == "trial_invalid_signal_quality"
        assert summary["score"] == 0.0

        import json

        rows = [json.loads(l) for l in events.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        judges = [r for r in rows if r["event"] == "judge"]
        assert judges and all(r.get("signal_bad") for r in judges)
        assert not any(r.get("score") for r in judges if r.get("score"))
