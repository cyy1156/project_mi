"""累积小考卷（爬坡曲线数据源）与准入门槛。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .readout import judge_trial


@dataclass
class QuizTrial:
    trial_id: int
    label: int
    judgment_windows: np.ndarray  # (n_j, 8, 750)：t=3/4/5/6 各判定窗
    game: bool = False


@dataclass
class CurvePoint:
    k_ft: int          # 累计微调试次数
    n_quiz: int        # 小考卷规模
    acc: float         # 试次级多数票准确率
    round_no: int


class QuizStore:
    """留出试次累积（6/12/18/24…）；试次永不进微调（隔离由调用方 + 断言保证）。"""

    def __init__(self) -> None:
        self.trials: List[QuizTrial] = []
        self._seen_ids: set = set()
        self.curve: List[CurvePoint] = []

    def add(self, trial: QuizTrial) -> None:
        if trial.trial_id in self._seen_ids:
            raise ValueError(f"quiz trial {trial.trial_id} 重复入库")
        self._seen_ids.add(trial.trial_id)
        self.trials.append(trial)

    def evaluate(self, predict_window: Callable[[np.ndarray], Dict], *, round_no: int = 0, k_ft: int = 0) -> CurvePoint:
        """predict_window(window) → {"pred": int, ...}（含串行门控的读数策略）。

        试次级判定 = 4 次秒判定多数票（judge_trial）。
        """
        n_ok = 0
        for t in self.trials:
            per_j = [predict_window(w) for w in t.judgment_windows]
            verdict = judge_trial(t.label, per_j)
            n_ok += int(verdict.correct)
        acc = n_ok / max(1, len(self.trials))
        pt = CurvePoint(k_ft=k_ft, n_quiz=len(self.trials), acc=float(acc), round_no=round_no)
        self.curve.append(pt)
        return pt

    @property
    def n_trials(self) -> int:
        return len(self.trials)


@dataclass
class GateDecision:
    status: str            # "pass" | "fail_pending" | "extend" | "weak_mi"
    acc: float
    n_quiz: int
    round_no: int


class AdmissionGate:
    """准入判定（常量驱动，现场禁改）。

    status 语义：
      pass         —— 小考卷 ≥ gate_min_quiz_trials 且 acc ≥ gate_enter_three
      fail_pending —— 尚未到起判规模（继续轮次）
      extend       —— 未达标但 round < cal_rounds_max
      weak_mi      —— 超上限仍未达标（标记，不硬调）
    """

    def __init__(self, constants) -> None:
        self.c = constants
        self.decisions: List[GateDecision] = []

    def update(self, acc: float, n_quiz: int, round_no: int) -> GateDecision:
        if n_quiz < self.c.gate_min_quiz_trials:
            status = "fail_pending"
        elif acc >= self.c.gate_enter_three:
            status = "pass"
        elif round_no < self.c.cal_rounds_max:
            status = "extend"
        else:
            status = "weak_mi"
        d = GateDecision(status=status, acc=float(acc), n_quiz=int(n_quiz), round_no=int(round_no))
        self.decisions.append(d)
        return d
