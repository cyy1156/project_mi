"""MI / Rest 试次计分：因果平滑后多数票（F5 单轨）。"""

from __future__ import annotations

from typing import Any, Dict, List

from experiment_game.experiment.judge_aggregate import (
    apply_causal_smooth_to_judgments,
    primary_judge_from_judgments,
)

# Cue 前静息多数票正确得分（与 Left/Right 各 18 分对齐：36×0.5=18）
PRE_CUE_REST_POINTS = 0.5
CAUSAL_LOOKBACK = 2


class MiTrialTracker:
    """F5：无 τ 早停；窗级因果平滑 argmax；试次多数票；Rest/MI 同规则。"""

    def __init__(self, label: int, *, correct_points: float = 1.0) -> None:
        self.label = int(label)
        self.correct_points = float(correct_points)
        self.windows: List[Dict[str, Any]] = []

    def add_window(self, t_rel: float, j: Dict[str, Any]) -> Dict[str, Any]:
        pred_raw = int(j.get("pred", 0))
        gated = bool(j.get("gated", False))
        gated_pred_raw = 0 if gated else pred_raw
        rec = {
            "t_rel": float(t_rel),
            "pred_raw": pred_raw,
            "pred": pred_raw,
            "gated_pred": gated_pred_raw,
            "gated": gated,
            "p_max": float(j.get("p_max", 0.0)),
            "p_three": j.get("p_three"),
            "win_start_rel": j.get("win_start_rel"),
            "win_end_rel": j.get("win_end_rel"),
        }
        self.windows.append(rec)
        # 用截至当前的因果平滑更新本窗展示/反馈 pred
        smoothed = apply_causal_smooth_to_judgments(
            self.windows, lookback=CAUSAL_LOOKBACK
        )
        if smoothed:
            last = smoothed[-1]
            rec["pred"] = int(last["pred"])
            rec["gated_pred"] = 0 if gated else int(last["pred"])
            rec["p_max"] = float(last["p_max"])
            rec["p_three_smooth"] = list(last["p_three"])
            rec["causal_smooth"] = True
        return rec

    def running_arm_score(self, *, cap: float = 5.0) -> float:
        """伸手反馈用：每窗（因果平滑后）判对 +1，上限 cap。"""
        correct = sum(1 for w in self.windows if w["gated_pred"] == self.label)
        return min(float(cap), float(correct))

    def finalize(self, *, signal_bad_trial: bool = False) -> Dict[str, Any]:
        base = {
            "label": self.label,
            "early_stop": False,
            "window_judgments": list(self.windows),
            "n_judgments": len(self.windows),
            "correct_points": self.correct_points,
        }
        if signal_bad_trial or not self.windows:
            return {
                **base,
                "score": 0.0,
                "valid": False,
                "correct": False,
                "pred": None,
                "rule": "causal_smooth_majority",
                "vote_counts": {},
                "invalid_reason": (
                    "trial_invalid_signal_quality" if signal_bad_trial else "no_judgments"
                ),
            }

        primary = primary_judge_from_judgments(
            self.windows, mode="majority", causal_lookback=CAUSAL_LOOKBACK
        )
        pred = int(primary["pred"]) if primary else -1
        correct = pred == self.label
        return {
            **base,
            "score": float(self.correct_points) if correct else 0.0,
            "valid": True,
            "correct": correct,
            "pred": pred,
            "rule": "causal_smooth_majority",
            "vote_counts": primary.get("vote_counts") if primary else {},
            "primary_judge": primary,
            "invalid_reason": None,
        }


def session_score_max_openbmi(
    n_mi_trials: int,
    *,
    inter_trial_rest_s: float = 4.0,
) -> float:
    """L/R 各 +1（满分 n_mi）；若有 Cue前静息，再 + n_mi×0.5（n_mi=36 → 54）。"""
    n = int(n_mi_trials)
    if float(inter_trial_rest_s) <= 1e-6:
        return float(n)
    return float(n) + float(n) * PRE_CUE_REST_POINTS


def empty_session_score_by() -> Dict[str, float]:
    """本场分项：Left / Right / Cue前静息（正式 Rest 标签仅指 Cue前静息）。"""
    return {"left": 0.0, "right": 0.0, "pre_cue_rest": 0.0}


def session_score_by_max(
    n_mi_trials: int,
    *,
    inter_trial_rest_s: float = 4.0,
) -> Dict[str, float]:
    """分项满分参考（L/R 按半场；Cue前静息 = n×0.5）。"""
    n = max(0, int(n_mi_trials))
    n_l = n // 2
    n_r = n - n_l
    cue_rest = float(n) * PRE_CUE_REST_POINTS if float(inter_trial_rest_s) > 1e-6 else 0.0
    return {
        "left": float(n_l),
        "right": float(n_r),
        "pre_cue_rest": cue_rest,
    }


def add_session_score_points(progress: Dict[str, Any], pts: float, *, bucket: str) -> None:
    """累加总分，并写入 session_score_by 分项。"""
    pts = float(pts or 0.0)
    progress["session_score"] = float(progress.get("session_score") or 0) + pts
    by = progress.get("session_score_by")
    if not isinstance(by, dict):
        by = empty_session_score_by()
        progress["session_score_by"] = by
    by[bucket] = float(by.get(bucket) or 0) + pts
