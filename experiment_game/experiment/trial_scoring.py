"""MI 试次计分：记录每窗类别，MI 结束用多数票；正确 +1，跑完全程即 valid。"""

from __future__ import annotations

from typing import Any, Dict, List

from experiment_game.experiment.judge_aggregate import primary_judge_from_judgments


class MiTrialTracker:
    """替代 D8 OnlineScoreTracker：无早停、无错类熔断。"""

    def __init__(self, label: int) -> None:
        self.label = int(label)
        self.windows: List[Dict[str, Any]] = []

    def add_window(self, t_rel: float, j: Dict[str, Any]) -> Dict[str, Any]:
        pred = int(j.get("pred", 0))
        gated = bool(j.get("gated", False))
        gated_pred = 0 if gated else pred
        rec = {
            "t_rel": float(t_rel),
            "pred": pred,
            "gated_pred": gated_pred,
            "gated": gated,
            "p_max": float(j.get("p_max", 0.0)),
            "p_three": j.get("p_three"),
            "win_start_rel": j.get("win_start_rel"),
            "win_end_rel": j.get("win_end_rel"),
        }
        self.windows.append(rec)
        return rec

    def running_arm_score(self, *, cap: float = 5.0) -> float:
        """伸手反馈用：每窗判对 +1，上限 cap（兼容 map_score_to_arm）。"""
        correct = sum(1 for w in self.windows if w["gated_pred"] == self.label)
        return min(float(cap), float(correct))

    def finalize(self, *, signal_bad_trial: bool = False) -> Dict[str, Any]:
        base = {
            "label": self.label,
            "early_stop": False,
            "window_judgments": list(self.windows),
            "n_judgments": len(self.windows),
        }
        if signal_bad_trial or not self.windows:
            return {
                **base,
                "score": 0.0,
                "valid": False,
                "correct": False,
                "pred": None,
                "rule": "majority_vote",
                "vote_counts": {},
                "invalid_reason": (
                    "trial_invalid_signal_quality" if signal_bad_trial else "no_judgments"
                ),
            }

        primary = primary_judge_from_judgments(self.windows, mode="majority")
        pred = int(primary["pred"]) if primary else -1
        correct = pred == self.label
        return {
            **base,
            "score": 1.0 if correct else 0.0,
            "valid": True,
            "correct": correct,
            "pred": pred,
            "rule": "majority_vote",
            "vote_counts": primary.get("vote_counts") if primary else {},
            "primary_judge": primary,
            "invalid_reason": None,
        }
