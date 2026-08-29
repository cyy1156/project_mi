"""试次判定聚合：因果平滑后多数票（F5 单轨）。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


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


def apply_causal_smooth_to_judgments(
    judgments: List[Dict[str, Any]],
    *,
    lookback: int = 2,
) -> List[Dict[str, Any]]:
    """复制判定列表，用因果平滑后的 argmax 覆盖 pred/p_max/p_three。"""
    from adapt_engine.readout import causal_smooth_pred_sequence

    js = [j for j in judgments if not j.get("signal_bad")]
    if not js:
        return []
    probs = []
    for j in js:
        p3 = j.get("p_three")
        if isinstance(p3, (list, tuple)) and len(p3) >= 3:
            probs.append(np.asarray(p3, dtype=np.float32))
        else:
            # 无概率时退化为 one-hot(pred)
            pred = int(j.get("gated_pred", j.get("pred", 0)))
            oh = np.zeros(3, dtype=np.float32)
            if 0 <= pred < 3:
                oh[pred] = 1.0
            else:
                oh[0] = 1.0
            probs.append(oh)
    smoothed = causal_smooth_pred_sequence(probs, lookback=lookback)
    out: List[Dict[str, Any]] = []
    for j, sm in zip(js, smoothed):
        rec = dict(j)
        rec["pred"] = int(sm["pred"])
        rec["gated_pred"] = int(sm["pred"])
        rec["p_max"] = float(sm["p_max"])
        rec["p_three"] = list(sm["p_three"])
        rec["causal_smooth"] = True
        out.append(rec)
    return out


def primary_judge_from_judgments(
    judgments: List[Dict[str, Any]],
    *,
    mode: str = "majority",
    primary_s: float = 4.0,
    e1f_smooth_radius: int = 1,
    e1f_tau_conf: float = 0.4,
    causal_lookback: int = 2,
) -> Optional[Dict[str, Any]]:
    """从判定窗列表得到试次主判定。

    mode:
      - majority / causal_majority: 因果平滑后多数票（F5 默认；平票取 p 之和最大）
      - nearest: 距 primary_s 最近的一档（legacy）
      - e1f_conf_stop: 历史旁路（正式 SOP 已退出）
    """
    if mode in ("majority", "causal_majority", ""):
        js_ok = apply_causal_smooth_to_judgments(
            judgments, lookback=causal_lookback
        )
        if not js_ok:
            return None
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
        out["rule"] = "causal_smooth_majority"
        out["vote_counts"] = {int(k): int(v) for k, v in cnt.items()}
        out["causal_lookback"] = int(causal_lookback)
        return out

    js_ok = [j for j in judgments if not j.get("signal_bad")]
    if not js_ok:
        return None

    if mode == "e1f_conf_stop":
        from adapt_engine.readout import e1f_conf_stop_from_judgments

        return e1f_conf_stop_from_judgments(
            judgments,
            smooth_radius=e1f_smooth_radius,
            tau_conf=e1f_tau_conf,
            primary_s=primary_s,
        )

    if mode == "nearest":
        return min(js_ok, key=lambda j: abs(float(j.get("t_rel", 0.0)) - float(primary_s)))

    # fallback
    return primary_judge_from_judgments(
        judgments, mode="majority", primary_s=primary_s, causal_lookback=causal_lookback
    )
