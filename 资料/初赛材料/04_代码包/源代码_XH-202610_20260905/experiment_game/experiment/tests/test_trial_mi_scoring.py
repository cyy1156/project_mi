"""MI 多数票计分（替代 D8）。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiment_game.experiment import trial_v2 as tv2  # noqa: E402
from experiment_game.experiment.events_log import EventLogger  # noqa: E402
from experiment_game.experiment.markers import MarkerPublisher  # noqa: E402
from experiment_game.experiment.trial_scoring import MiTrialTracker  # noqa: E402
from experiment_game.experiment.trial_v2 import (  # noqa: E402
    TrialContextV2,
    TrialStateMachineV2,
    TrialTimingV2,
)


def _noop_wait(t_end, **kwargs) -> None:
    return None


def test_mi_tracker_majority_correct():
    tr = MiTrialTracker(1)
    for t_rel, pred in [(3.0, 1), (3.1, 1), (3.2, 2), (4.0, 1)]:
        tr.add_window(t_rel, {"pred": pred, "gated": False, "p_max": 0.8})
    s = tr.finalize()
    assert s["valid"] is True
    assert s["correct"] is True
    assert s["score"] == 1.0
    assert s["pred"] == 1
    assert s["rule"] == "causal_smooth_majority"


def test_mi_tracker_causal_smooth_uses_p_three():
    """因果平滑可改变单窗 argmax，从而影响多数票。"""
    tr = MiTrialTracker(1)
    # 窗0: 偏 Rest；窗1–2: 强 Left → 平滑后应稳住 Left
    windows = [
        (3.0, [0.5, 0.3, 0.2], 0),
        (3.1, [0.1, 0.8, 0.1], 1),
        (3.2, [0.1, 0.8, 0.1], 1),
    ]
    for t_rel, p3, pred in windows:
        tr.add_window(
            t_rel,
            {"pred": pred, "gated": False, "p_max": max(p3), "p_three": p3},
        )
    s = tr.finalize()
    assert s["pred"] == 1
    assert s["correct"] is True
    assert s["primary_judge"]["rule"] == "causal_smooth_majority"


def test_mi_tracker_majority_wrong():
    tr = MiTrialTracker(1)
    for pred in [2, 2, 2, 1]:
        tr.add_window(3.0, {"pred": pred, "gated": False, "p_max": 0.8})
    s = tr.finalize()
    assert s["valid"] is True
    assert s["correct"] is False
    assert s["score"] == 0.0


def test_mi_tracker_rest_majority_correct():
    tr = MiTrialTracker(0)
    for pred in [0, 0, 1, 0]:
        tr.add_window(3.0, {"pred": pred, "gated": False, "p_max": 0.8})
    s = tr.finalize()
    assert s["valid"] is True
    assert s["correct"] is True
    assert s["score"] == 1.0
    assert s["pred"] == 0


def test_mi_tracker_pre_cue_rest_half_point():
    from experiment_game.experiment.trial_scoring import PRE_CUE_REST_POINTS, session_score_max_openbmi

    tr = MiTrialTracker(0, correct_points=PRE_CUE_REST_POINTS)
    for pred in [0, 0, 1, 0]:
        tr.add_window(3.0, {"pred": pred, "gated": False, "p_max": 0.8})
    s = tr.finalize()
    assert s["correct"] is True
    assert s["score"] == 0.5
    assert session_score_max_openbmi(36) == 54.0
    assert session_score_max_openbmi(36, inter_trial_rest_s=0) == 36.0


def test_session_score_by_buckets():
    from experiment_game.experiment.trial_scoring import (
        add_session_score_points,
        empty_session_score_by,
        session_score_by_max,
    )

    assert session_score_by_max(36) == {
        "left": 18.0,
        "right": 18.0,
        "pre_cue_rest": 18.0,
    }
    prog = {"session_score": 0.0, "session_score_by": empty_session_score_by()}
    add_session_score_points(prog, 1.0, bucket="left")
    add_session_score_points(prog, 0.5, bucket="pre_cue_rest")
    add_session_score_points(prog, 1.0, bucket="right")
    assert prog["session_score"] == 2.5
    assert prog["session_score_by"] == {
        "left": 1.0,
        "right": 1.0,
        "pre_cue_rest": 0.5,
    }


def test_scoring_replay_skips_pre_cue_rest_judges():
    import json
    import tempfile
    from pathlib import Path

    from experiment_game.experiment.scoring_replay import load_judge_rows, replay_session

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "events.jsonl"
        rows = [
            {"event": "trial_start", "trial_id": 1, "label": 1},
            {
                "event": "judge",
                "trial_id": 1,
                "t_rel": 3.0,
                "pred": 0,
                "gated": False,
                "score_phase": "pre_cue_rest",
                "role": "pre_cue_rest",
            },
            {
                "event": "judge",
                "trial_id": 1,
                "t_rel": 3.0,
                "pred": 1,
                "gated": False,
                "score_phase": "mi",
            },
            {
                "event": "judge",
                "trial_id": 1,
                "t_rel": 3.1,
                "pred": 1,
                "gated": False,
            },
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        mi = load_judge_rows(p, score_phase="mi")
        assert len(mi[1]) == 2
        assert all(j["pred"] == 1 for j in mi[1])
        rest = load_judge_rows(p, score_phase="pre_cue_rest")
        assert len(rest[1]) == 1
        out = replay_session(p)
        assert len(out) == 1
        assert out[0]["correct"] is True
        assert out[0]["score"] == 1.0
        assert out[0]["rest_score"] == 0.5


def test_pre_cue_rest_scores_half():
    tv2._wu = _noop_wait
    times = tuple(round(3.0 + i * 0.1, 1) for i in range(11))
    timing = TrialTimingV2(
        prep_s=0.001,
        cue_s=0.0,
        imagine_s=4.0,
        iti_s=0.001,
        inter_trial_rest_s=4.0,
        judgment_times=times,
    )
    with tempfile.TemporaryDirectory() as td:
        events = EventLogger(Path(td) / "events.jsonl")
        markers = MarkerPublisher(enabled=False)
        n_rest_judge = {"n": 0}

        def judgment_fn(anchor_t, t_rel, ctx):
            if getattr(ctx, "score_phase", "mi") == "pre_cue_rest":
                n_rest_judge["n"] += 1
                return {"pred": 0, "p_max": 0.9, "gated": False}
            return {"pred": 1, "p_max": 0.9, "gated": False}

        sm = TrialStateMachineV2(
            events,
            markers,
            timing,
            judgment_fn=judgment_fn,
        )
        ctx = TrialContextV2(trial_id=1, label=1, mode="calibration", round_no=1)
        summary = sm.run_inter_trial_rest(ctx)
        events.close()
        assert summary is not None
        assert summary["correct"] is True
        assert summary["score"] == 0.5
        assert summary["role"] == "pre_cue_rest"
        assert n_rest_judge["n"] == len(times)


def test_trial_runs_full_mi_no_early_stop():
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
        events = EventLogger(Path(td) / "events.jsonl")
        markers = MarkerPublisher(enabled=False)
        n_judge = {"n": 0}

        def judgment_fn(mi_t, t_rel, ctx):
            n_judge["n"] += 1
            return {"pred": 1, "p_max": 0.9, "gated": False}

        sm = TrialStateMachineV2(
            events,
            markers,
            timing,
            judgment_fn=judgment_fn,
        )
        ctx = TrialContextV2(trial_id=1, label=1, mode="calibration", round_no=1)
        summary = sm.run_trial(ctx)
        events.close()

        assert summary is not None
        assert summary["valid"] is True
        assert summary["score"] == 1.0
        assert summary["early_stop"] is False
        assert n_judge["n"] == len(times)
        import json

        rows = [json.loads(l) for l in events.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        ev_names = [e["event"] for e in rows]
        assert "score_reach" not in ev_names
        assert "trial_invalid" not in ev_names


def test_mi_judgment_wait_is_mi_relative_when_cue_s_gt_0():
    """cue_s>0 时判定等待锚点必须是 mi_t（MI 段起止切窗），不能是 cue_t。"""
    waits: list[tuple[float, float]] = []

    def capture_wait(self, t_start, duration_s):
        waits.append((float(t_start), float(duration_s)))

    tv2._wu = _noop_wait
    orig = tv2.TrialStateMachineV2._wait_after
    tv2.TrialStateMachineV2._wait_after = capture_wait
    try:
        timing = TrialTimingV2(
            prep_s=0.0,
            cue_s=1.0,
            imagine_s=4.0,
            iti_s=0.0,
            inter_trial_rest_s=0.0,
            judgment_times=(3.0, 4.0),
        )
        with tempfile.TemporaryDirectory() as td:
            events = EventLogger(Path(td) / "events.jsonl")
            markers = MarkerPublisher(enabled=False)
            cue_t_box = {"t": None}
            mi_t_box = {"t": None}

            def judgment_fn(mi_t, t_rel, ctx):
                mi_t_box["t"] = float(mi_t)
                return {"pred": 1, "p_max": 0.9, "gated": False}

            def on_stage(stage, ctx, data=None):
                if stage == "cue" and isinstance(data, dict):
                    cue_t_box["t"] = float(data["cue_t"])
                if stage == "mi_start" and isinstance(data, dict):
                    mi_t_box["t"] = float(data["mi_t"])

            sm = TrialStateMachineV2(
                events,
                markers,
                timing,
                judgment_fn=judgment_fn,
                on_stage=on_stage,
            )
            ctx = TrialContextV2(trial_id=1, label=1, mode="calibration", round_no=1)
            sm.run_trial(ctx)
            events.close()

        assert cue_t_box["t"] is not None and mi_t_box["t"] is not None
        cue_t = cue_t_box["t"]
        mi_t = mi_t_box["t"]
        # _wait_after 被 stub 时墙钟不前进；改为断言 Cue 段等待 (cue_t, cue_s=1)
        assert any(abs(a - cue_t) < 1e-6 and abs(d - 1.0) < 1e-9 for a, d in waits)
        # 必须存在相对 mi 的判定等待；不得用 cue 锚 3.0s 判定
        assert any(abs(a - mi_t) < 1e-6 and abs(d - 3.0) < 1e-9 for a, d in waits)
        assert any(abs(a - mi_t) < 1e-6 and abs(d - 4.0) < 1e-9 for a, d in waits)
        assert not any(abs(a - cue_t) < 1e-6 and abs(d - 3.0) < 1e-9 for a, d in waits)
    finally:
        tv2.TrialStateMachineV2._wait_after = orig
        tv2._wu = _noop_wait