"""trial_v2 MI 多数票计分冒烟（无真实等待 / 无 LSL）。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiment_game.experiment import trial_v2 as tv2  # noqa: E402
from experiment_game.experiment.events_log import EventLogger  # noqa: E402
from experiment_game.experiment.markers import MarkerPublisher  # noqa: E402
from experiment_game.experiment.trial_v2 import (  # noqa: E402
    TrialContextV2,
    TrialStateMachineV2,
    TrialTimingV2,
)
from adapt_engine.scoring_v21 import build_judgment_times  # noqa: E402


def _fast_timing() -> TrialTimingV2:
    return TrialTimingV2(
        prep_s=0.001,
        cue_s=0.001,
        imagine_s=6.0,
        iti_s=0.001,
        inter_trial_rest_s=0.0,
        judgment_times=build_judgment_times(0.6, 6.0),
    )


def _noop_wait(t_end, **kwargs) -> None:
    return None


def _read_events(path: Path) -> list:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_trial_runs_full_mi_no_early_stop():
    """错类全程仍跑完 MI；多数票错 → score=0，无 score_reach。"""
    tv2._wu = _noop_wait
    with tempfile.TemporaryDirectory() as td:
        events = EventLogger(Path(td) / "events.jsonl")
        markers = MarkerPublisher(enabled=False)
        label = 1

        def judgment_fn(mi_t, t_rel, ctx):
            return {"pred": 2, "p_max": 0.9, "gated": False}

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
        assert summary["correct"] is False
        assert summary["score"] == 0.0
        assert summary["early_stop"] is False
        ev_names = [e["event"] for e in _read_events(events.path)]
        assert "score_reach" not in ev_names
        assert "trial_invalid" not in ev_names
        mi_end = [e for e in _read_events(events.path) if e["event"] == "mi_end"]
        assert mi_end and mi_end[0].get("early") is False


def test_majority_correct_scores_one():
    tv2._wu = _noop_wait
    with tempfile.TemporaryDirectory() as td:
        events = EventLogger(Path(td) / "events.jsonl")
        markers = MarkerPublisher(enabled=False)
        label = 1
        calls = {"n": 0}

        def judgment_fn(mi_t, t_rel, ctx):
            calls["n"] += 1
            return {"pred": label, "p_max": 0.9, "gated": False}

        sm = TrialStateMachineV2(
            events,
            markers,
            _fast_timing(),
            judgment_fn=judgment_fn,
        )
        ctx = TrialContextV2(trial_id=1, label=label, mode="calibration", round_no=1)
        summary = sm.run_trial(ctx)
        events.close()

        assert summary["score"] == 1.0
        assert summary["correct"] is True
        assert calls["n"] == len(_fast_timing().judgment_times)
