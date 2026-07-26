"""正式采集试次状态机（控制器主导计时，无画面）。"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from pylsl import local_clock

from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.markers import MarkerPublisher, format_payload
from experiment_game.experiment.timing import DEFAULT_TIMING, TrialTiming


OnStage = Callable[[str, "TrialContext", Optional[int]], None]
_MISSING = object()


def wait_until(
    t_end: float,
    *,
    is_paused: Optional[Callable[[], bool]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> None:
    """等到 local_clock() >= t_end；暂停时冻结终点；abort 时抛 SessionAbort。"""
    while True:
        if should_abort is not None and should_abort():
            raise SessionAbort("operator abort")
        if is_paused is not None and is_paused():
            t0 = local_clock()
            while is_paused():
                if should_abort is not None and should_abort():
                    raise SessionAbort("operator abort")
                time.sleep(0.02)
            t_end += local_clock() - t0
            continue
        now = local_clock()
        remaining = t_end - now
        if remaining <= 0:
            break
        time.sleep(min(remaining, 0.02))


class SessionAbort(Exception):
    """操作者紧急结束会话。"""
    pass


def build_label_schedule(n_trials: int, rng: Optional[random.Random] = None) -> List[int]:
    """
    同物品左右配对：每 2 trial 为 {1,2} 随机顺序。
    n_trials 为奇数时最后一 trial 单独随机 1/2。
    """
    rng = rng or random.Random()
    labels: List[int] = []
    pairs = n_trials // 2
    for _ in range(pairs):
        pair = [1, 2]
        rng.shuffle(pair)
        labels.extend(pair)
    if n_trials % 2 == 1:
        labels.append(rng.choice([1, 2]))
    return labels


@dataclass
class TrialContext:
    trial_id: int
    label: int
    object: str
    scene: str
    phase: str = "acquire"
    transition_amp: str = "micro"  # micro | swap | scene
    rejected: bool = False
    learn_step: Optional[int] = None


class TrialStateMachine:
    def __init__(
        self,
        events: EventLogger,
        markers: MarkerPublisher,
        timing: TrialTiming = DEFAULT_TIMING,
        on_stage: Optional[OnStage] = None,
        is_paused: Optional[Callable[[], bool]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
        is_rejected: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.events = events
        self.markers = markers
        self.timing = timing
        self.on_stage = on_stage
        self.is_paused = is_paused
        self.should_abort = should_abort
        self.is_rejected = is_rejected

    def _wait(self, t_end: float) -> None:
        wait_until(
            t_end,
            is_paused=self.is_paused,
            should_abort=self.should_abort,
        )

    def _emit(
        self,
        event: str,
        ctx: TrialContext,
        *,
        label: object = _MISSING,
        extra: Optional[dict] = None,
    ) -> dict:
        lab: Optional[int]
        if label is _MISSING:
            lab = ctx.label
        else:
            lab = label  # type: ignore[assignment]
        payload = format_payload(
            event,
            trial_id=ctx.trial_id,
            label=lab,
            phase=ctx.phase,
            object_name=ctx.object,
            scene=ctx.scene,
            learn_step=ctx.learn_step,
        )
        fields = {
            "trial_id": ctx.trial_id,
            "phase": ctx.phase,
            "object": ctx.object,
            "scene": ctx.scene,
            "payload": payload,
        }
        if ctx.learn_step is not None:
            fields["learn_step"] = ctx.learn_step
        if lab is not None:
            fields["label"] = lab
        if extra:
            fields.update(extra)
        row = self.events.emit(event, **fields)
        self.markers.push(payload, t_lsl=row["t_lsl"])
        return row

    def _notify(self, stage: str, ctx: TrialContext, label: object = _MISSING) -> None:
        if self.on_stage is not None:
            if label is _MISSING:
                lab: Optional[int] = ctx.label
            else:
                lab = label  # type: ignore[assignment]
            self.on_stage(stage, ctx, lab)

    def run_trial(self, ctx: TrialContext) -> None:
        t = self.timing
        self._notify("trial_start", ctx)
        self._emit("trial_start", ctx)

        # Fixation
        self._notify("fixation", ctx, label=None)
        row = self._emit("fixation", ctx, label=None)
        self._wait(row["t_lsl"] + t.fixation_s)

        # Cue
        self._notify("cue", ctx)
        row = self._emit("cue", ctx)
        self._wait(row["t_lsl"] + t.cue_s)

        # MI
        self._notify("mi", ctx)
        row = self._emit("mi_start", ctx)
        self._wait(row["t_lsl"] + t.mi_s)
        self._emit("mi_end", ctx)

        # PostMI hold（不单独打点）
        self._notify("post_mi_hold", ctx)
        hold_t0 = local_clock()
        self._wait(hold_t0 + t.post_mi_hold_s)

        # Rest
        self._notify("rest", ctx, label=0)
        row = self._emit("rest_start", ctx, label=0)
        self._wait(row["t_lsl"] + t.rest_s)
        self._emit("rest_end", ctx, label=0)

        # Transition
        self._notify("transition", ctx, label=None)
        row = self._emit(
            "transition",
            ctx,
            label=None,
            extra={"transition_amp": ctx.transition_amp},
        )
        self._wait(row["t_lsl"] + t.transition_s)

        if self.is_rejected is not None and self.is_rejected():
            ctx.rejected = True
        if ctx.rejected:
            self._emit(
                "trial_reject",
                ctx,
                extra={"reason": "operator_reject"},
            )

        self._emit("trial_end", ctx)
        self._notify("trial_end", ctx)

    def run_block(
        self,
        n_trials: int,
        *,
        object_name: str = "cup",
        scene: str = "home_desk",
        phase: str = "acquire",
        labels: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> List[int]:
        rng = random.Random(seed)
        schedule = list(labels) if labels is not None else build_label_schedule(n_trials, rng)
        if len(schedule) != n_trials:
            raise ValueError("labels 长度必须等于 n_trials")

        self.events.emit("phase_start", phase=phase, trial_id=None, label=None)
        self.markers.push(format_payload("phase_start", phase=phase))

        for i, lab in enumerate(schedule, start=1):
            ctx = TrialContext(
                trial_id=i,
                label=int(lab),
                object=object_name,
                scene=scene,
                phase=phase,
            )
            self.run_trial(ctx)

        self.events.emit("phase_end", phase=phase, trial_id=None, label=None)
        self.markers.push(format_payload("phase_end", phase=phase))
        return schedule
