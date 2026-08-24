"""读出策略（三可替换接口之二）。

默认：串行门控 + 均匀秒判定多数票。
扩展点：24-V 置信/时间加权（换 score_fn 即可）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def serial_gating(
    p_task: np.ndarray,
    p_three: np.ndarray,
    *,
    task_p_on: float = 0.6,
) -> Dict:
    """串行门控：Task 头 P(task) < task_p_on → 直接静息；否则用 Three 头 argmax。

    p_task: (2,) [P(rest), P(task)]（约定 0=静息）。
    返回 {"pred": 0/1/2, "p_max": float, "gated": bool}。
    """
    p_task_on = float(p_task[1]) if len(p_task) == 2 else 1.0 - float(p_task[0])
    if p_task_on < task_p_on:
        return {"pred": 0, "p_max": float(1.0 - p_task_on), "gated": True}
    pred = int(np.argmax(p_three))
    return {"pred": pred, "p_max": float(np.max(p_three)), "gated": False}


@dataclass
class TrialVerdict:
    label: int
    preds: List[int]              # 各判定点预测（含门控静息）
    correct_per_j: List[bool]
    n_correct: int
    majority_pred: int
    correct: bool                 # 多数票对（≥3/4）
    reach: Optional[bool] = None  # 游戏模式：是否 4 档到位
    reach_time: Optional[float] = None


def judge_trial(
    label: int,
    per_judgment: Sequence[Dict],
    *,
    n_levels: int = 4,
) -> TrialVerdict:
    """试次级判定：per_judgment = [{pred, p_max, t}, ...]（t=3/4/5/6s 顺序）。

    多数票（含平票取先出现类）；游戏模式下 reach = 连续正确累计到位。
    """
    preds = [int(j["pred"]) for j in per_judgment]
    n = len(preds)
    votes: Dict[int, int] = {}
    for p in preds:
        votes[p] = votes.get(p, 0) + 1
    majority_pred = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    correct_per = [p == int(label) for p in preds]
    n_correct = sum(correct_per)

    reach = None
    reach_time = None
    if n > 0 and per_judgment[0].get("is_game"):
        level = 0
        for j, ok in zip(per_judgment, correct_per):
            if ok:
                level += 1
                if level >= n_levels:
                    reach = True
                    reach_time = float(j.get("t", 0.0))
                    break
        if reach is None:
            reach = False

    return TrialVerdict(
        label=int(label),
        preds=preds,
        correct_per_j=correct_per,
        n_correct=n_correct,
        majority_pred=int(majority_pred),
        correct=(majority_pred == int(label)),
        reach=reach,
        reach_time=reach_time,
    )


def confidence_weighted_majority(per_judgment: Sequence[Dict]) -> int:
    """24-V 扩展位：按 p_max 加权投票（阳性后替换均匀多数票）。"""
    scores: Dict[int, float] = {}
    for j in per_judgment:
        scores[int(j["pred"])] = scores.get(int(j["pred"]), 0.0) + float(j.get("p_max", 1.0))
    return int(max(scores.items(), key=lambda kv: kv[1])[0])
