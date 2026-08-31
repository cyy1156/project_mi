"""v2 游戏测试引擎（基于 v3 最终权重 · 20 试次 · 满分 15）。

试次结构（需求 2026-08-30）：
  - 静息试次（label=0，10 个）：仅 Rest 5s，全程计时、不提前结束；
    连续 5 窗判定为静息 → +0.5 分。
  - MI 试次（label=1/2，各 5 个）：Rest 5s 引入（不计分）→ Cue 1s
    （文字「接下来请你想象用左手/右手拿前面的杯子」）→ MI 10s；
    连续 5 窗识别出对应手 → 手拿杯子（cup_grasp）、MI 提前结束、+1 分；
    未成功则跑满 10s、不得分。

判定流：每 ``judge_interval_s``（默认 0.5s）取一窗；「连续 N 窗」= 连续 N 次
判定（默认 5）同为目标标签，signal_bad 窗重置连击。
计分：10×0.5 + 10×1 = 15。模型权重 = 被试 v3 最终权重（current/members + overlay）。
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from pylsl import local_clock

from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.markers import MarkerPublisher, format_payload
from experiment_game.experiment.trial_sm import SessionAbort, wait_until
from experiment_game.experiment.trial_v2 import (
    LABEL_LEFT,
    LABEL_REST,
    LABEL_RIGHT,
    TrialContextV2,
)

GAME_SCORE_MAX_FALLBACK = 15.0


def build_v3test_schedule(
    n_rest: int,
    n_left: int,
    n_right: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """20 试次排程：10 静息 + 5 左 + 5 右，打散且同型不连续 ≥4。"""
    rng = rng or random.Random()
    base = [LABEL_REST] * int(n_rest) + [LABEL_LEFT] * int(n_left) + [LABEL_RIGHT] * int(n_right)
    for _ in range(200):
        rng.shuffle(base)
        if all(not (base[i] == base[i + 1] == base[i + 2] == base[i + 3]) for i in range(len(base) - 3)):
            return base
    return base


class StreakCounter:
    """连续命中计数：喂入判定，命中目标 +1，否则清零。"""

    def __init__(self, target: int, need: int = 5) -> None:
        self.target = int(target)
        self.need = max(1, int(need))
        self.count = 0
        self.max_count = 0

    def feed(self, pred: Optional[int]) -> int:
        if pred is not None and int(pred) == self.target:
            self.count += 1
        else:
            self.count = 0
        self.max_count = max(self.max_count, self.count)
        return self.count

    @property
    def hit(self) -> bool:
        return self.count >= self.need


def run_v3test_game(
    events: EventLogger,
    markers: MarkerPublisher,
    bridge,  # WsBridge
    on_console: Callable[[str], None],
    *,
    cfg,  # V2Config（读取 v3test_* 字段与 iti_s）
    judgment_fn: Optional[Callable[[float, float, TrialContextV2], Optional[Dict]]],
    on_stage: Callable[[str, TrialContextV2, Optional[dict]], None],
    is_paused: Optional[Callable[[], bool]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    seed: Optional[int] = None,
    n_trials_done_start: int = 0,
) -> Dict:
    """主循环。返回 summary（含 score / trials 明细）。

    ``judgment_fn(t0, t_rel, ctx)``：t0 为相位锚点（Rest=rest 起点；
    MI=cue 起点，与 OpenBMI 训练对齐一致），t_rel 为相对锚点的秒数。
    计分与累计进度由调用方 on_stage（session_v2.on_stage）完成：
    静息试次结束发 ``pre_cue_rest_end``（score=0.5/0），MI 试次结束发
    ``trial_end``（score=1/0，label∈{1,2}）。
    """
    import time as _time

    rng = random.Random(seed) if seed is not None else random.Random()
    rest_s = float(getattr(cfg, "v3test_rest_s", 5.0))
    cue_s = float(getattr(cfg, "v3test_cue_s", 1.0))
    mi_s = float(getattr(cfg, "v3test_mi_s", 10.0))
    iv = float(getattr(cfg, "v3test_judge_interval_s", 0.5) or 0.5)
    need = int(getattr(cfg, "v3test_consecutive", 5))
    rest_pts = float(getattr(cfg, "v3test_rest_points", 0.5))
    mi_pts = float(getattr(cfg, "v3test_mi_points", 1.0))
    n_rest = int(getattr(cfg, "v3test_n_rest", 10))
    n_left = int(getattr(cfg, "v3test_n_left", 5))
    n_right = int(getattr(cfg, "v3test_n_right", 5))
    score_max = n_rest * rest_pts + (n_left + n_right) * mi_pts

    schedule = build_v3test_schedule(n_rest, n_left, n_right, rng)
    rest_times = [round(iv * i, 3) for i in range(1, int(round(rest_s / iv)) + 1) if iv * i <= rest_s + 1e-9]
    mi_times = [
        round(cue_s + iv * i, 3)
        for i in range(1, int(round((cue_s + mi_s) / iv)) + 1)
        if cue_s + iv * i <= cue_s + mi_s + 1e-9
    ]

    def _wait(t_end: float) -> None:
        while local_clock() < t_end:
            if should_abort is not None and should_abort():
                raise SessionAbort("operator_abort")
            if is_paused is not None and is_paused():
                _time.sleep(0.1)
                continue
            _time.sleep(0.02)

    def _wait_after(t0: float, dur: float) -> None:
        _wait(float(t0) + float(dur))

    def _emit(event: str, ctx: TrialContextV2, extra: Optional[dict] = None) -> dict:
        payload = format_payload(event, trial_id=ctx.trial_id, label=ctx.label, phase=ctx.mode)
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
        row = events.emit(event, **fields)
        markers.push(payload, t_lsl=row["t_lsl"])
        return row

    on_console(
        f"[v2·game] v3 权重游戏测试：{len(schedule)} 试次"
        f"（静息{n_rest}×{rest_pts} + MI{n_left + n_right}×{mi_pts}，满分 {score_max:g}）；"
        f"Rest {rest_s:g}s / Cue {cue_s:g}s / MI {mi_s:g}s，连击门槛 {need} 窗"
    )
    stub = TrialContextV2(trial_id=0, label=0, mode="game", round_no=1)
    on_stage("round_start", stub, {"mode": "game", "round": 1, "game_mode": "v3_test"})
    events.emit("round_start", phase="game", round=1, payload=format_payload("round_start", phase="game"))
    markers.push("round_start|phase=game")

    trial_results: List[Dict] = []
    score_total = 0.0
    aborted = False
    trial_id = n_trials_done_start

    try:
        for idx, label in enumerate(schedule):
            trial_id += 1
            ctx = TrialContextV2(
                trial_id=trial_id, label=int(label), mode="game", round_no=1,
                score_phase="mi" if label != LABEL_REST else "pre_cue_rest",
            )
            on_stage("trial_start", ctx, {"trial_index": idx + 1, "trial_total": len(schedule)})
            _emit("trial_start", ctx)

            # —— Rest 5s（全时长；静息试次计分，MI 试次仅热身）——
            scores_rest = label == LABEL_REST
            streak_rest = StreakCounter(LABEL_REST, need=need)
            on_stage("rest_start", ctx, {
                "duration_s": rest_s, "role": "v3test_rest",
                "rest_text": "保持静息",
            })
            row = _emit("rest_start", ctx, extra={"label": 0, "role": "v3test_rest"})
            rest_t = row["t_lsl"]
            rest_hit = False
            for t_rel in rest_times:
                # _wait_after(t0, dur) = 等到 t0+dur；须传相对 rest_t 的绝对偏移 t_rel
                _wait_after(rest_t, float(t_rel))
                if judgment_fn is None:
                    continue
                rest_ctx = TrialContextV2(
                    trial_id=trial_id, label=LABEL_REST, mode="game", round_no=1,
                    score_phase="pre_cue_rest",
                )
                j = judgment_fn(rest_t, float(t_rel), rest_ctx)
                if j is None or j.get("signal_bad"):
                    streak_rest.feed(None)
                    continue
                streak_rest.feed(j.get("pred"))
                j2 = dict(j)
                j2["t_rel"] = float(t_rel)
                j2["score_phase"] = "pre_cue_rest"
                j2["role"] = "v3test_rest"
                j2["streak"] = streak_rest.count
                j2["score"] = float(streak_rest.count)
                _emit("judge", rest_ctx, extra={
                    "t_rel": float(t_rel), "pred": j.get("pred"),
                    "p_max": round(float(j.get("p_max", 0.0)), 4),
                    "streak": streak_rest.count,
                    "score_phase": "pre_cue_rest", "role": "v3test_rest",
                })
                on_stage("judge", rest_ctx, j2)
                if scores_rest and streak_rest.hit and not rest_hit:
                    rest_hit = True
            _wait_after(rest_t, rest_s)
            _emit("rest_end", ctx, extra={"role": "v3test_rest", "rest_hit": rest_hit})

            if label == LABEL_REST:
                pts = rest_pts if rest_hit else 0.0
                score_total += pts
                summary = {
                    "label": LABEL_REST, "score": pts, "valid": True,
                    "hit": rest_hit, "streak_max": streak_rest.max_count,
                    "rule": "v3test_consecutive_rest", "role": "v3test_rest",
                }
                _emit("pre_cue_rest_end", ctx, extra={"trial_score": summary, "score": pts})
                on_stage("pre_cue_rest_end", ctx, {"summary": summary})
                trial_results.append({"trial_id": trial_id, "label": int(label), **summary})
                on_stage("trial_end", ctx, {"summary": {**summary, "score": 0.0}})
                _emit("trial_end", ctx, extra={"trial_score": {**summary, "score": 0.0}})
                continue

            # —— Cue（默认 1s，与 Align 一致）——
            side = "左手" if label == LABEL_LEFT else "右手"
            # cue 阶段：与 MI 同义指导语，最前面加 cue（游戏测试用长句）
            mi_line = f"接下来请你想象用{side}拿前面的杯子"
            cue_text = f"cue · {mi_line}"
            on_stage("cue", ctx, {"cue_text": cue_text, "cue_s": cue_s})
            row = _emit("cue", ctx, extra={"cue_kind": "v3test", "cue_text": cue_text})
            cue_t = row["t_lsl"]
            _wait_after(cue_t, cue_s)

            # —— MI 10s：连击达标 → 拿杯、提前结束、+1 —— 
            streak = StreakCounter(int(label), need=need)
            on_stage("mi", ctx, {"mi_s": mi_s, "cue_text": mi_line})
            row = _emit("mi_start", ctx)
            mi_t = row["t_lsl"]
            on_stage("mi_start", ctx, {"mi_t": mi_t, "cue_t": cue_t, "mi_end_t": mi_t + mi_s})
            success = False
            for t_rel in mi_times:
                # mi_times 为相对 cue_t 的绝对偏移（含 cue_s 段）；勿传相邻差，否则判定瞬间爆发
                _wait_after(cue_t, float(t_rel))
                if judgment_fn is None:
                    continue
                j = judgment_fn(cue_t, float(t_rel), ctx)
                if j is None:
                    continue
                if j.get("signal_bad"):
                    streak.feed(None)
                    bad = dict(j)
                    bad.update({"t_rel": float(t_rel), "score_phase": "mi", "streak": streak.count})
                    _emit("judge", ctx, extra={
                        "t_rel": float(t_rel), "signal_bad": True,
                        "signal_reason": j.get("reason"), "score_phase": "mi",
                        "streak": streak.count,
                    })
                    on_stage("judge", ctx, bad)
                    continue
                streak.feed(j.get("pred"))
                j2 = dict(j)
                j2["t_rel"] = float(t_rel)
                j2["score_phase"] = "mi"
                j2["streak"] = streak.count
                j2["score"] = float(streak.count)
                j2["arm_level"] = min(4, streak.count)
                j2["cup_grasp"] = streak.hit
                _emit("judge", ctx, extra={
                    "t_rel": float(t_rel), "pred": j.get("pred"),
                    "p_max": round(float(j.get("p_max", 0.0)), 4),
                    "streak": streak.count, "score_phase": "mi",
                    "cup_grasp": streak.hit,
                })
                on_stage("judge", ctx, j2)
                if streak.hit and not success:
                    success = True
                    _emit("mi_success", ctx, extra={
                        "t_rel": float(t_rel), "streak": streak.count,
                    })
                    on_console(
                        f"[v2·game] trial {trial_id}: {'左' if label == LABEL_LEFT else '右'}手连击 {streak.count} 窗达标 → 拿杯 +{mi_pts:g}"
                    )
                    break  # MI 提前结束
            mi_early = bool(success)
            if not success:
                _wait_after(cue_t, cue_s + mi_s)  # 跑满 MI（锚点=cue 起点）
            _emit("mi_end", ctx, extra={"early": mi_early, "reason": "success" if success else "full_mi"})

            pts = mi_pts if success else 0.0
            score_total += pts
            summary = {
                "label": int(label), "score": pts, "valid": True,
                "hit": success, "streak_max": streak.max_count,
                "early_stop": mi_early, "rule": "v3test_consecutive_mi",
            }
            _emit("trial_end", ctx, extra={"trial_score": summary})
            on_stage("trial_end", ctx, {"summary": summary})
            trial_results.append({"trial_id": trial_id, "label": int(label), **summary})

            # —— ITI ——
            on_stage("iti", ctx)
            row = _emit("iti_start", ctx)
            _wait_after(row["t_lsl"], float(getattr(cfg, "iti_s", 3.0)))
    except SessionAbort as exc:
        aborted = True
        on_console(f"[v2·game] 会话中止：{exc}")
        _emit("game_abort", stub, extra={"reason": str(exc)})

    on_stage("game_end", stub, {"score": round(score_total, 2), "score_max": score_max})
    events.emit(
        "game_end", phase="game",
        score=round(score_total, 2), score_max=score_max,
        n_trials=len(trial_results), aborted=aborted,
        payload=format_payload("game_end", phase="game"),
    )
    markers.push(f"game_end|score={score_total:g}")

    return {
        "game_mode": "v3_test",
        "gate_status": "skipped_v3test",
        "rounds": 1,
        "curve": [],
        "degraded": judgment_fn is None,
        "aborted": aborted,
        "abort_reason": "operator_abort" if aborted else None,
        "score": round(score_total, 2),
        "score_max": score_max,
        "n_trials": len(trial_results),
        "trials": trial_results,
        "drift_stats": [],
        "skips": {"guidance": True, "calibration": True, "gate": True, "game": False},
    }
