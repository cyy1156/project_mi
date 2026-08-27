"""试次判定聚合：MI 全程多数票等。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


def _sum_p_three(judgments: List[Dict[str, Any]], pred: int) -> float:
    total = 0.0
    for j in judgments:
        if int(j.get("pred", -1)) != pred:
            continue
        p3 = j.get("p_three")
        if isinstance(p3, (list, tuple)) and len(p3) > pred:
            total += float(p3[pred])
        elif j.get("p_max") is not None:
            total += float(j["p_max"])
    return total


def primary_judge_from_judgments(
    judgments: List[Dict[str, Any]],
    *,
    mode: str = "majority",
    primary_s: float = 4.0,
) -> Optional[Dict[str, Any]]:
    """从判定窗列表得到试次主判定。

    mode:
      - majority: 非 signal_bad 窗多数票；平票取 p_three 之和最大类
      - nearest: 距 primary_s 最近的一档（legacy）
    """
    js_ok = [j for j in judgments if not j.get("signal_bad")]
    if not js_ok:
        return None

    if mode == "nearest":
        return min(js_ok, key=lambda j: abs(float(j.get("t_rel", 0.0)) - float(primary_s)))

    preds = [int(j.get("gated_pred", j.get("pred", 0))) for j in js_ok]
    cnt = Counter(preds)
    top = cnt.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        winner = top[0][0]
    else:
        tied = [p for p, n in top if n == top[0][1]]
        winner = max(tied, key=lambda p: _sum_p_three(js_ok, p))

    cand = [j for j in js_ok if int(j.get("gated_pred", j.get("pred", 0))) == winner]
    rep = min(cand, key=lambda j: abs(float(j.get("t_rel", 0.0)) - float(primary_s)))
    out = dict(rep)
    out["pred"] = winner
    out["rule"] = "majority_vote"
    out["vote_counts"] = {int(k): int(v) for k, v in cnt.items()}
    return out
