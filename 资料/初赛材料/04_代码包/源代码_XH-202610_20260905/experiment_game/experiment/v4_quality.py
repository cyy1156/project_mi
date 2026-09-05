"""v4 连续质量监控：streak 与历史聚合。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiment_game.experiment.v4_config import V4Config


class V4QualityMonitor:
    def __init__(self, cfg: V4Config) -> None:
        self.cfg = cfg
        self.streak = 0
        self.window_idx = 0
        self.achieved_stable = False
        self.time_to_stable_s: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
        self._pass_announced = False

    def rolling_verdict(self) -> str:
        if self.achieved_stable:
            return "pass"
        if self.streak > 0 and self.streak < self.cfg.pass_streak_required:
            return "warn"
        if self.history and self.history[-1].get("window_ok"):
            return "warn"
        return "fail"

    def update(self, diag: Dict[str, Any], *, elapsed_s: float) -> Optional[Dict[str, Any]]:
        """记录一窗；若刚达成连续达标则返回 v4_pass 载荷。"""
        self.window_idx += 1
        rec = dict(diag)
        rec["window_idx"] = self.window_idx
        rec["elapsed_s"] = round(elapsed_s, 2)
        rec["pass_streak"] = self.streak
        rec["rolling_verdict"] = self.rolling_verdict()
        self.history.append(rec)

        if diag.get("window_ok"):
            self.streak += 1
        else:
            self.streak = 0

        rec["pass_streak"] = self.streak
        rec["rolling_verdict"] = self.rolling_verdict()

        pass_evt: Optional[Dict[str, Any]] = None
        if (
            not self._pass_announced
            and self.streak >= self.cfg.pass_streak_required
        ):
            self.achieved_stable = True
            self.time_to_stable_s = elapsed_s
            self._pass_announced = True
            n = self.cfg.pass_streak_required
            sec = n * self.cfg.eval_interval_s
            pass_evt = {
                "streak": self.streak,
                "elapsed_s": round(elapsed_s, 2),
                "message": f"信号稳定：连续 {n} 窗（{sec:.0f}s）达标，可以开始 v3 实验",
            }
        return pass_evt
