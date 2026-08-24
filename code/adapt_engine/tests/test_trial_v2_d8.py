"""trial_v2 D8 接线冒烟（无真实等待 / 无 LSL）。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiment_game" / ".."))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiment_game.experiment import trial_v2 as tv2  # noqa: E402
from experiment_game.experiment.events_log import EventLogger  # noqa: E402
from experiment_game.experiment.markers import MarkerPublisher  # noqa: E402
from experiment_game.experiment.trial_sm import SessionAbort  # noqa: E402
from experiment_game.experiment.trial_v2 import (  # noqa: E402
    TrialContextV2,
    TrialStateMachineV2,
    TrialTimingV2,
)
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


def _noop_wait(t_end, **kwargs) -> None:
    return None


def _read_events(path: Path) -> list:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_trial_early_stop_score_reach():
    tv2._wu = _noop_wait
    with tempfile.TemporaryDirectory() as td:
        events = EventLogger(Path(td) / "events.jsonl")
        markers = MarkerPublisher(enabled=False)
        label = 1

        def judgment_fn(mi_t, t_rel, ctx):
            return {"pred": label, "p_max": 0.9, "gated": False}

        sm = TrialStateMachineV2(
            events,
            markers,
            _fast_timing(),
            judgment_fn=judgment_fn,
        )
        ctx = TrialContextV2(trial_id=1, label=label, mode="game", round_no=1)
        summary = sm.run_trial(ctx)
        events.close()

        assert summary is not None
        assert summary["valid"] is True
        assert summary["early_stop"] is True
        ev_names = [e["event"] for e in _read_events(events.path)]
        assert "score_reach" in ev_names
        assert "reach" not in ev_names


def test_consecutive_invalid_abort():
    tv2._wu = _noop_wait
    with tempfile.TemporaryDirectory() as td:
        events = EventLogger(Path(td) / "events.jsonl")
        markers = MarkerPublisher(enabled=False)
        aborts = []

        def on_trial_end(ctx, summary):
            if summary and not summary.get("valid"):
                aborts.append(ctx.trial_id)
                if len(aborts) >= 5:
                    return "abort_session"
            return None

        def judgment_fn(mi_t, t_rel, ctx):
            return {"pred": 2, "p_max": 0.9, "gated": False}

        sm = TrialStateMachineV2(
            events,
            markers,
            _fast_timing(),
            judgment_fn=judgment_fn,
            on_trial_end=on_trial_end,
        )
        raised = False
        for i in range(6):
            ctx = TrialContextV2(trial_id=i + 1, label=1, mode="calibration", round_no=1)
            try:
                sm.run_trial(ctx)
            except SessionAbort:
                raised = True
                break
        events.close()
        assert raised
        assert len(aborts) == 5
