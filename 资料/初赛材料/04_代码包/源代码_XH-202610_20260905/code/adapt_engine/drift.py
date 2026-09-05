"""漂移保护（场内熔断）：连续 N 轮不升反降 → 回滚 + lr 减半；再触发 → 冻结。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class DriftAction(Enum):
    NONE = "none"
    ROLLBACK_LR = "rollback_lr"     # 一档：回滚上一 ckpt + lr 减半
    FREEZE = "freeze"               # 二档：冻结在线更新


@dataclass
class DriftRecord:
    round_no: int
    metric: float
    action: DriftAction
    note: str = ""


class DriftGuard:
    """指标 = 轮内（游戏）单秒平均准确率，或轮末小考精度（标定）。

    用法（与 IncrementalFinetuner 配合）：
        guard = DriftGuard(...)
        guard.before_round(fin.save_checkpoint("pre"))   # 存 ckpt
        ... 轮结束 ...
        action = guard.after_round(round_no, metric, rollback_fn=fin.rollback,
                                   halve_fn=fin.halve_lr, state=last_state)
    """

    def __init__(self, patience: int = 2, *, min_rounds: int = 2) -> None:
        self.patience = int(patience)
        self.min_rounds = int(min_rounds)
        self.history: List[DriftRecord] = []
        self._metrics: List[float] = []
        self._frozen: bool = False
        self._last_action_round: int = -10

    @property
    def frozen(self) -> bool:
        return self._frozen

    def before_round(self, save_ckpt_fn) -> Optional[object]:
        """轮前存档；save_ckpt_fn() 返回 ckpt 句柄/路径。冻结态不再存。"""
        if self._frozen:
            return None
        return save_ckpt_fn()

    def after_round(
        self,
        round_no: int,
        metric: float,
        *,
        rollback_fn=None,
        halve_fn=None,
        get_state_fn=None,
    ) -> DriftAction:
        self._metrics.append(float(metric))
        action = DriftAction.NONE
        note = ""

        if self._frozen:
            action = DriftAction.FREEZE
            note = "已冻结，跳过更新"
        elif (
            round_no >= self.min_rounds
            and round_no - self._last_action_round >= self.patience
        ):
            # 「连续 patience 组不升反降」= patience 次下降 = 最近 patience+1 个点单调递减
            recent = self._metrics[-(self.patience + 1) :]
            if len(recent) == self.patience + 1 and all(
                recent[i + 1] < recent[i] - 1e-9 for i in range(len(recent) - 1)
            ):
                if self._already_rolled_once():
                    action = DriftAction.FREEZE
                    self._frozen = True
                    note = "二次触发：冻结在线更新"
                else:
                    action = DriftAction.ROLLBACK_LR
                    if get_state_fn is not None and rollback_fn is not None:
                        rollback_fn(get_state_fn())
                    if halve_fn is not None:
                        halve_fn()
                    self._rollback_done = True
                    note = "一档：回滚 + lr 减半"
                self._last_action_round = round_no

        self.history.append(
            DriftRecord(round_no=round_no, metric=float(metric), action=action, note=note)
        )
        return action

    def _already_rolled_once(self) -> bool:
        return getattr(self, "_rollback_done", False)

    def trend(self) -> List[float]:
        return list(self._metrics)
