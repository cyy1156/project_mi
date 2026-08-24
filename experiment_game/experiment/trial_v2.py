"""v2 试次引擎（6s 范式 + 标定/游戏双模式 + D8 加权判定）。

范式（v2.1 · D8）：
  0–2s 准备 → 2s Cue → mi_start → MI ≤6s（0.6s 栅格判分）→ ITI 3s
  早停：真标签加权分 ≥5；错类加权 ≥5 → 无效；否则满 6s 后 Score≤3 无效。
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from pylsl import local_clock

from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.markers import MarkerPublisher, format_payload
from experiment_game.experiment.trial_sm import SessionAbort, wait_until as _wu  # noqa: F401

_ADAPT_ENGINE_ROOT = str(Path(__file__).resolve().parents[2] / "code")
if _ADAPT_ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ADAPT_ENGINE_ROOT)

from adapt_engine.scoring_v21 import OnlineScoreTracker, ScoringConfig, build_judgment_times  # noqa: E402

LABEL_REST, LABEL_LEFT, LABEL_RIGHT = 0, 1, 2
CUE_KIND = {LABEL_LEFT: "anim_ball_grasp", LABEL_RIGHT: "anim_writing", LABEL_REST: "icon_rest"}


@dataclass(frozen=True)
class TrialTimingV2:
    prep_s: float = 2.0
    cue_s: float = 2.0
    imagine_s: float = 6.0
    iti_s: float = 3.0
    judgment_times: tuple = ()
    scoring: ScoringConfig = ScoringConfig()

    def __post_init__(self):
        if not self.judgment_times:
            object.__setattr__(
                self,
                "judgment_times",
                build_judgment_times(self.scoring.judgment_step_s, self.imagine_s),
            )

    @property
    def total_s(self) -> float:
        return self.prep_s + self.cue_s + self.imagine_s + self.iti_s


DEFAULT_TIMING_V2 = TrialTimingV2()


# —— 排程器 ——
def _permute_subblock(rng: random.Random) -> List[int]:
    block = [LABEL_LEFT, LABEL_LEFT, LABEL_RIGHT, LABEL_RIGHT, LABEL_REST, LABEL_REST]
    for _ in range(50):
        rng.shuffle(block)
        if all(not (block[i] == block[i + 1] == block[i + 2]) for i in range(len(block) - 2)):
            return block
    return block


def build_calibration_schedule(rng: Optional[random.Random] = None) -> List[int]:
    rng = rng or random.Random()
    out: List[int] = []
    for _ in range(3):
        out.extend(_permute_subblock(rng))
    return out


def build_game_schedule(n: int = 16, rng: Optional[random.Random] = None) -> List[int]:
    rng = rng or random.Random()
    base = [LABEL_LEFT] * (n // 3 + (1 if n % 3 >= 1 else 0))
    base += [LABEL_RIGHT] * (n // 3 + (1 if n % 3 >= 2 else 0))
    base += [LABEL_REST] * (n - len(base))
    for _ in range(50):
        rng.shuffle(base)
        if all(not (base[i] == base[i + 1] == base[i + 2]) for i in range(len(base) - 2)):
            return base
    return base


@dataclass
class TrialContextV2:
    trial_id: int
    label: int
    mode: str
    round_no: int
    rejected: bool = False
    subblock: int = 0


TrialEndHook = Callable[[TrialContextV2, Optional[Dict]], Optional[str]]


class TrialStateMachineV2:
    """judgment_fn(mi_t, t_rel, ctx) → {pred, p_max, gated, ...}。"""

    def __init__(
        self,
        events: EventLogger,
        markers: MarkerPublisher,
        timing: TrialTimingV2 = DEFAULT_TIMING_V2,
        *,
        on_stage: Optional[Callable[[str, TrialContextV2, Optional[dict]], None]] = None,
        judgment_fn: Optional[Callable[[float, float, TrialContextV2], Optional[Dict]]] = None,
        on_trial_end: Optional[TrialEndHook] = None,
        is_paused: Optional[Callable[[], bool]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
        is_rejected: Optional[Callable[[], bool]] = None,
        touch_pending: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.events = events
        self.markers = markers
        self.timing = timing
        self.on_stage = on_stage
        self.judgment_fn = judgment_fn
        self.on_trial_end = on_trial_end
        self.is_paused = is_paused
        self.should_abort = should_abort
        self.is_rejected = is_rejected
        self.touch_pending = touch_pending

    def _wait(self, t_end: float) -> None:
        _wu(t_end, is_paused=self.is_paused, should_abort=self.should_abort)

    def _emit(self, event: str, ctx: TrialContextV2, extra: Optional[dict] = None) -> dict:
        payload = format_payload(
            event, trial_id=ctx.trial_id, label=ctx.label, phase=ctx.mode
        )
        fields = {
            "trial_id": ctx.trial_id,
            "phase": ctx.mode,
            "round": ctx.round_no,
            "payload": payload,
        }
        if ctx.label is not None:
            fields["label"] = ctx.label
        if extra:
            fields.update(extra)
        row = self.events.emit(event, **fields)
        self.markers.push(payload, t_lsl=row["t_lsl"])
        return row

    def _notify(self, stage: str, ctx: TrialContextV2, data: Optional[dict] = None) -> None:
        if self.on_stage is not None:
            self.on_stage(stage, ctx, data)

    def run_trial(self, ctx: TrialContextV2) -> Optional[Dict]:
        t = self.timing
        sc = t.scoring
        self._notify("trial_start", ctx)
        self._emit("trial_start", ctx, extra={"subblock": ctx.subblock})

        self._notify("prep", ctx)
        row = self._emit("prep_start", ctx)
        self._wait(row["t_lsl"] + t.prep_s)

        self._notify("cue", ctx)
        row = self._emit("cue", ctx, extra={"cue_kind": CUE_KIND.get(ctx.label, "icon_rest")})
        cue_t = row["t_lsl"]
        self._wait(cue_t + t.cue_s)

        self._notify("mi", ctx)
        row = self._emit("mi_start", ctx)
        mi_t = row["t_lsl"]
        self._notify("mi_start", ctx, {"mi_t": mi_t})

        tracker = OnlineScoreTracker(ctx.label, sc)
        mi_end_early = False
        end_reason: Optional[str] = None
        signal_bad_ticks = 0

        for t_rel in t.judgment_times:
            if mi_end_early:
                break
            if self.touch_pending is not None and self.touch_pending():
                mi_end_early = True
                end_reason = "touch"
                self._emit("touch", ctx, extra={"t_rel": t_rel})
                break
            self._wait(mi_t + t_rel)
            if self.judgment_fn is None:
                continue
            j = self.judgment_fn(mi_t, t_rel, ctx)
            if j is None:
                continue
            if j.get("signal_bad"):
                signal_bad_ticks += 1
                self._emit(
                    "judge", ctx,
                    extra={
                        "t_rel": t_rel,
                        "signal_bad": True,
                        "signal_reason": j.get("reason"),
                        "signal_metrics": j.get("signal_metrics"),
                    },
                )
                self._notify("judge", ctx, j)
                continue
            j = dict(j)
            tick = tracker.apply_tick(t_rel, int(j.get("pred", 0)), extra={
                "p_max": j.get("p_max"),
                "gated": j.get("gated"),
            })
            j.update(tick)
            j["t_rel"] = t_rel
            j["is_game"] = ctx.mode == "game"
            self._emit(
                "judge", ctx,
                extra={
                    "t_rel": t_rel,
                    "pred": j.get("pred"),
                    "p_max": round(float(j.get("p_max", 0.0)), 4),
                    "gated": bool(j.get("gated", False)),
                    "weight": j.get("weight"),
                    "score": round(float(j.get("score", 0.0)), 4),
                },
            )
            self._notify("judge", ctx, j)

            if tick.get("invalid_wrong_race"):
                mi_end_early = True
                end_reason = "trial_invalid_wrong_race"
                break
            if tick.get("early_stop"):
                mi_end_early = True
                end_reason = "score_5"
                self._emit("score_reach", ctx, extra={"t_rel": t_rel, "score": tick["score"]})
                self._notify("score_reach", ctx, {"t_rel": t_rel, "score": tick["score"]})
                break

        if not mi_end_early:
            self._wait(mi_t + t.imagine_s)
            end_reason = end_reason or "full_6s"

        self._emit("mi_end", ctx, extra={"early": mi_end_early, "reason": end_reason})

        self._notify("iti", ctx)
        row = self._emit("iti_start", ctx)
        self._wait(row["t_lsl"] + t.iti_s)

        if self.is_rejected is not None and self.is_rejected():
            ctx.rejected = True
        if ctx.rejected:
            self._emit("trial_reject", ctx, extra={"reason": "operator_reject"})

        summary_dict: Optional[Dict] = None
        if tracker.judgments:
            verdict = tracker.finalize(ended_early=mi_end_early, end_reason=end_reason)
            summary_dict = verdict.to_dict()
            if verdict.invalid and not ctx.rejected:
                self._emit(
                    "trial_invalid", ctx,
                    extra={"reason": verdict.invalid_reason, "score": verdict.score},
                )
        elif signal_bad_ticks > 0 and not ctx.rejected:
            summary_dict = {
                "label": ctx.label,
                "score": 0.0,
                "valid": False,
                "invalid_reason": "trial_invalid_signal_quality",
                "signal_bad_ticks": signal_bad_ticks,
                "early_stop": False,
            }
            self._emit(
                "trial_invalid", ctx,
                extra={
                    "reason": "trial_invalid_signal_quality",
                    "score": 0.0,
                    "signal_bad_ticks": signal_bad_ticks,
                },
            )
        self._emit("trial_end", ctx, extra={"score_v21": summary_dict})
        self._notify("trial_end", ctx, {"summary": summary_dict})

        if self.on_trial_end is not None and not ctx.rejected:
            action = self.on_trial_end(ctx, summary_dict)
            if action == "abort_session":
                raise SessionAbort("consecutive_invalid_abort")

        return summary_dict

    def run_round(
        self,
        labels: Sequence[int],
        *,
        mode: str,
        round_no: int,
        trial_id_offset: int = 0,
    ) -> List[int]:
        stub = TrialContextV2(
            trial_id=0, label=0, mode=mode, round_no=round_no, subblock=0,
        )
        self._notify("round_start", stub, {"mode": mode, "round": round_no})
        self.events.emit("round_start", phase=mode, round=round_no, payload=format_payload("round_start", phase=mode))
        self.markers.push(format_payload("round_start", phase=mode))
        per_sub = 6 if mode == "calibration" else len(labels)
        for i, lab in enumerate(labels):
            ctx = TrialContextV2(
                trial_id=trial_id_offset + i + 1,
                label=int(lab),
                mode=mode,
                round_no=round_no,
                subblock=(i // per_sub) + 1 if mode == "calibration" else 0,
            )
            self.run_trial(ctx)
        self._notify("round_end", stub, {"mode": mode, "round": round_no})
        self.events.emit("round_end", phase=mode, round=round_no, payload=format_payload("round_end", phase=mode))
        self.markers.push(format_payload("round_end", phase=mode))
        return list(labels)


def run_guidance_stage(
    events: EventLogger,
    markers: MarkerPublisher,
    *,
    round_no: int,
    on_stage: Optional[Callable[[str, dict], None]] = None,
    confirm_fn: Optional[Callable[[], bool]] = None,
    timeout_s: float = 600.0,
) -> dict:
    t0 = local_clock()
    events.emit("guidance_begin", round=round_no, payload=format_payload("guidance_begin"))
    markers.push(format_payload("guidance_begin"))
    if on_stage:
        on_stage("guidance_begin", {"round": round_no})
    passed = False
    if confirm_fn is not None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if confirm_fn():
                passed = True
                break
            time.sleep(0.2)
    dur = local_clock() - t0
    events.emit("guidance_end", round=round_no, passed=passed,
                duration_s=round(dur, 2), payload=format_payload("guidance_end"))
    markers.push(format_payload("guidance_end"))
    if on_stage:
        on_stage("guidance_end", {"round": round_no, "passed": passed, "duration_s": dur})
    return {"round": round_no, "passed": passed, "duration_s": dur}
