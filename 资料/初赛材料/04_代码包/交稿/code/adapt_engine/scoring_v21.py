"""v2.1 在线判定计分（D8 · experiment_game v2_upgrade_plan §3.4）。

栅格：mi_start 后 t_k = step·k（默认 step=0.6s）至 imagine_s。
票权：t_k ≤ half_until → 0.5；否则 1.0（对/错对称）。
早停：真标签加权分 ≥ score_early_stop（默认 5.0）。
无效：任错类加权 ≥ wrong_class_abort；或终局 score ≤ score_invalid_max。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ScoringConfig:
    judgment_step_s: float = 0.6
    judgment_half_weight_until_s: float = 2.4
    imagine_s: float = 6.0
    score_early_stop: float = 5.0
    score_invalid_max: float = 3.0
    score_valid_min: float = 4.0
    wrong_class_abort: float = 5.0


def build_judgment_times(step_s: float, imagine_s: float) -> Tuple[float, ...]:
    """mi_start 后判定点：step, 2·step, … ≤ imagine_s。"""
    step = float(step_s)
    if step <= 0:
        raise ValueError("judgment_step_s must be positive")
    out: List[float] = []
    t = step
    while t <= float(imagine_s) + 1e-9:
        out.append(round(t, 6))
        t += step
    return tuple(out)


def tick_weight(t_rel: float, half_until_s: float) -> float:
    return 0.5 if float(t_rel) <= float(half_until_s) + 1e-9 else 1.0


@dataclass
class TrialScoreV21:
    label: int
    score: float
    wrong_scores: Dict[int, float] = field(default_factory=dict)
    judgments: List[Dict] = field(default_factory=list)
    valid: bool = True
    invalid_reason: Optional[str] = None
    early_stop: bool = False
    early_stop_reason: Optional[str] = None
    early_stop_t_rel: Optional[float] = None

    @property
    def invalid(self) -> bool:
        return not self.valid

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "wrong_scores": {int(k): round(v, 4) for k, v in self.wrong_scores.items()},
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "early_stop": self.early_stop,
            "early_stop_reason": self.early_stop_reason,
            "early_stop_t_rel": self.early_stop_t_rel,
            "n_judgments": len(self.judgments),
            "n_correct_ticks": sum(1 for j in self.judgments if j.get("correct")),
        }


class OnlineScoreTracker:
    """试次内逐档累计；支持早停与错类抢先作废。"""

    def __init__(self, label: int, cfg: ScoringConfig):
        self.label = int(label)
        self.cfg = cfg
        self.score = 0.0
        self.wrong_scores: Dict[int, float] = {}
        self.judgments: List[Dict] = []

    def apply_tick(self, t_rel: float, pred: int, extra: Optional[Dict] = None) -> Dict:
        w = tick_weight(t_rel, self.cfg.judgment_half_weight_until_s)
        pred = int(pred)
        correct = pred == self.label
        if correct:
            self.score += w
        else:
            self.wrong_scores[pred] = self.wrong_scores.get(pred, 0.0) + w

        row = {
            "t": float(t_rel),
            "pred": pred,
            "weight": w,
            "correct": correct,
            "score": self.score,
            "wrong_scores": dict(self.wrong_scores),
        }
        if extra:
            row.update(extra)
        self.judgments.append(row)

        early_stop = self.score >= self.cfg.score_early_stop
        wrong_race = any(v >= self.cfg.wrong_class_abort for v in self.wrong_scores.values())
        return {
            **row,
            "early_stop": early_stop,
            "invalid_wrong_race": wrong_race,
        }

    def finalize(self, *, ended_early: bool, end_reason: Optional[str]) -> TrialScoreV21:
        wrong_race = any(v >= self.cfg.wrong_class_abort for v in self.wrong_scores.values())
        if wrong_race:
            return TrialScoreV21(
                label=self.label,
                score=self.score,
                wrong_scores=dict(self.wrong_scores),
                judgments=list(self.judgments),
                valid=False,
                invalid_reason="trial_invalid_wrong_race",
                early_stop=ended_early,
                early_stop_reason=end_reason,
                early_stop_t_rel=_last_t(self.judgments),
            )
        valid = self.score > self.cfg.score_invalid_max
        invalid_reason = None
        if not valid:
            invalid_reason = "trial_invalid_low_score"
        early = ended_early and end_reason == "score_5"
        return TrialScoreV21(
            label=self.label,
            score=self.score,
            wrong_scores=dict(self.wrong_scores),
            judgments=list(self.judgments),
            valid=valid,
            invalid_reason=invalid_reason if not valid else None,
            early_stop=early,
            early_stop_reason=end_reason if early else None,
            early_stop_t_rel=_last_t(self.judgments) if early else None,
        )


def score_trial_from_judgments(
    label: int,
    per_judgment: Sequence[Dict],
    cfg: ScoringConfig,
    *,
    ended_early: bool = False,
    end_reason: Optional[str] = None,
) -> TrialScoreV21:
    """离线回放：按时间顺序重放判定列表。"""
    tracker = OnlineScoreTracker(label, cfg)
    last_early = False
    last_wrong = False
    for j in sorted(per_judgment, key=lambda x: float(x.get("t", 0.0))):
        tick = tracker.apply_tick(float(j["t"]), int(j["pred"]), extra={k: v for k, v in j.items()
                                                                        if k not in ("t", "pred")})
        last_early = tick["early_stop"]
        last_wrong = tick["invalid_wrong_race"]
        if last_early or last_wrong:
            break
    reason = end_reason
    if last_wrong:
        reason = "trial_invalid_wrong_race"
    elif last_early:
        reason = "score_5"
    return tracker.finalize(ended_early=ended_early or last_early or last_wrong, end_reason=reason)


def _last_t(judgments: List[Dict]) -> Optional[float]:
    return float(judgments[-1]["t"]) if judgments else None
