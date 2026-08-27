"""v2 试次引擎（OpenBMI-Align v1 · MI 多数票计分 + 全程跑完）。

范式（OpenBMI-Align v1）：
  【每试次 Cue 前 4s Rest】→ prep → Cue=mi_start → MI 固定 4s → ITI
  无 label=0「静息想象」试次；静息 = 块前 30s seed + 每试次 Cue 前 4s。
  计分：每窗记录 pred；MI 结束多数票 vs label → 正确 +1 / 错误 0。
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
from experiment_game.experiment.trial_scoring import MiTrialTracker
from experiment_game.experiment.trial_sm import SessionAbort, wait_until as _wu  # noqa: F401

_ADAPT_ENGINE_ROOT = str(Path(__file__).resolve().parents[2] / "code")
if _ADAPT_ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ADAPT_ENGINE_ROOT)

LABEL_REST, LABEL_LEFT, LABEL_RIGHT = 0, 1, 2
CUE_KIND = {
    LABEL_LEFT: "anim_ball_grasp",
    LABEL_RIGHT: "anim_ball_grasp",
    LABEL_REST: "icon_rest",
}


@dataclass(frozen=True)
class TrialTimingV2:
    prep_s: float = 2.0
    cue_s: float = 0.0
    imagine_s: float = 4.0
    iti_s: float = 3.0
    inter_trial_rest_s: float = 4.0
    judgment_times: tuple = ()

    @property
    def total_s(self) -> float:
        # 含每试次 Cue 前 Rest
        return (
            self.inter_trial_rest_s
            + self.prep_s
            + self.cue_s
            + self.imagine_s
            + self.iti_s
        )


DEFAULT_TIMING_V2 = TrialTimingV2()


# —— 排程器（仅 Left/Right；无 Rest 想象试次）——
def _permute_lr_subblock(rng: random.Random) -> List[int]:
    """6 试次：3L+3R，避免同侧连续 ≥3。"""
    block = [LABEL_LEFT] * 3 + [LABEL_RIGHT] * 3
    for _ in range(50):
        rng.shuffle(block)
        if all(not (block[i] == block[i + 1] == block[i + 2]) for i in range(len(block) - 2)):
            return block
    return block


def build_calibration_schedule(rng: Optional[random.Random] = None) -> List[int]:
    """标定/探针块：默认 18 = 3×(3L+3R)。"""
    rng = rng or random.Random()
    out: List[int] = []
    for _ in range(3):
        out.extend(_permute_lr_subblock(rng))
    return out


def build_game_schedule(n: int = 16, rng: Optional[random.Random] = None) -> List[int]:
    """游戏轮：仅 Left/Right 均衡，无 Rest 试次。"""
    rng = rng or random.Random()
    n_left = n // 2
    n_right = n - n_left
    base = [LABEL_LEFT] * n_left + [LABEL_RIGHT] * n_right
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
        time_scale: float = 1.0,
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
        self.touch_pending = touch_pending  # legacy hook; 不再触发 MI 早停
        self.time_scale = max(1e-6, float(time_scale))

    def _wait(self, t_end: float) -> None:
        _wu(t_end, is_paused=self.is_paused, should_abort=self.should_abort)

    def _wait_after(self, t_start: float, duration_s: float) -> None:
        self._wait(float(t_start) + float(duration_s) * self.time_scale)

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
        self._notify("trial_start", ctx)
        self._emit("trial_start", ctx, extra={"subblock": ctx.subblock})

        self._notify("prep", ctx)
        row = self._emit("prep_start", ctx)
        self._wait_after(row["t_lsl"], t.prep_s)

        row = self._emit("cue", ctx, extra={"cue_kind": CUE_KIND.get(ctx.label, "icon_rest")})
        cue_t = row["t_lsl"]
        self._notify("cue", ctx, {"cue_t": cue_t})

        if t.cue_s > 1e-6:
            self._wait_after(cue_t, t.cue_s)

        self._notify("mi", ctx)
        row = self._emit("mi_start", ctx)
        mi_t = row["t_lsl"]
        self._notify("mi_start", ctx, {"mi_t": mi_t, "cue_t": cue_t})

        tracker = MiTrialTracker(ctx.label)
        signal_bad_ticks = 0
        good_ticks = 0

        for t_rel in t.judgment_times:
            self._wait_after(mi_t, t_rel)
            if self.judgment_fn is None:
                continue
            j = self.judgment_fn(mi_t, t_rel, ctx)
            if j is None:
                continue
            if j.get("signal_bad"):
                signal_bad_ticks += 1
                bad_payload = {
                    "t_rel": t_rel,
                    "signal_bad": True,
                    "signal_reason": j.get("reason"),
                    "signal_metrics": j.get("signal_metrics"),
                    "p_three": None,
                    "pred": None,
                }
                self._emit("judge", ctx, extra=bad_payload)
                self._notify("judge", ctx, {**j, **bad_payload})
                continue

            good_ticks += 1
            j = dict(j)
            from experiment_game.experiment.class_labels import normalize_p_three

            p_three = normalize_p_three(j.get("p_three"))
            j["p_three"] = p_three
            win = tracker.add_window(t_rel, j)
            j.update(win)
            j["t_rel"] = t_rel
            j["is_game"] = ctx.mode == "game"
            j["score"] = tracker.running_arm_score()
            self._emit(
                "judge",
                ctx,
                extra={
                    "t_rel": t_rel,
                    "pred": j.get("pred"),
                    "gated_pred": win.get("gated_pred"),
                    "p_max": round(float(j.get("p_max", 0.0)), 4),
                    "gated": bool(j.get("gated", False)),
                    "p_three": p_three,
                    "win_start_rel": j.get("win_start_rel"),
                    "win_end_rel": j.get("win_end_rel"),
                },
            )
            self._notify("judge", ctx, j)

        self._wait_after(mi_t, t.imagine_s)
        end_reason = "full_mi"

        self._emit("mi_end", ctx, extra={"early": False, "reason": end_reason})

        self._notify("iti", ctx)
        row = self._emit("iti_start", ctx)
        self._wait_after(row["t_lsl"], t.iti_s)

        if self.is_rejected is not None and self.is_rejected():
            ctx.rejected = True
        if ctx.rejected:
            self._emit("trial_reject", ctx, extra={"reason": "operator_reject"})

        signal_bad_trial = bool(good_ticks == 0 and signal_bad_ticks > 0)
        summary_dict: Optional[Dict] = tracker.finalize(signal_bad_trial=signal_bad_trial)
        if signal_bad_trial and not ctx.rejected:
            self._emit(
                "trial_invalid",
                ctx,
                extra={
                    "reason": summary_dict.get("invalid_reason"),
                    "score": summary_dict.get("score"),
                    "signal_bad_ticks": signal_bad_ticks,
                },
            )

        self._emit("trial_end", ctx, extra={
            "trial_score": summary_dict,
            "score_v21": summary_dict,
        })
        self._notify("trial_end", ctx, {"summary": summary_dict})

        if self.on_trial_end is not None and not ctx.rejected:
            self.on_trial_end(ctx, summary_dict)

        return summary_dict

    def run_inter_trial_rest(self, ctx: TrialContextV2) -> None:
        """每试次 Cue 前 Rest（OpenBMI 间隔段 / ERD 基线）；事件挂在本试次 trial_id。"""
        dur = self.timing.inter_trial_rest_s
        if dur <= 1e-6:
            return
        self._notify("inter_trial_rest", ctx)
        row = self._emit("rest_start", ctx, extra={"label": 0, "role": "pre_cue_rest"})
        rest_t = row["t_lsl"]
        self._notify("rest_start", ctx, {"rest_t": rest_t, "duration_s": dur})
        self._wait_after(rest_t, dur)
        self._emit("rest_end", ctx)
        self._notify("rest_end", ctx, {"rest_t": rest_t, "duration_s": dur})

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
            # 每试次（含块内首试）Cue 前 4s 静息；不跑 label=0 Rest 想象试次
            if self.timing.inter_trial_rest_s > 1e-6:
                self.run_inter_trial_rest(ctx)
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
