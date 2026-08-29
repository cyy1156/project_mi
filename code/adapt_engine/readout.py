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


def _smooth_prob_sequence(probs: Sequence[np.ndarray], radius: int) -> List[np.ndarray]:
    if radius <= 0:
        return [np.asarray(p, dtype=np.float32) for p in probs]
    out: List[np.ndarray] = []
    n = len(probs)
    for j in range(n):
        lo = max(0, j - radius)
        hi = min(n, j + radius + 1)
        sl = probs[lo:hi]
        out.append(np.mean(np.stack(sl, axis=0), axis=0).astype(np.float32))
    return out


def e1f_conf_stop_from_judgments(
    judgments: Sequence[Dict],
    *,
    smooth_radius: int = 1,
    tau_conf: float = 0.4,
    primary_s: float = 4.0,
) -> Optional[Dict]:
    """E1f 试次判定：窗级融合概率 → 邻域平滑 → p_max≥tau 早停。"""
    js_ok = [j for j in judgments if not j.get("signal_bad")]
    if not js_ok:
        return None
    js_ok = sorted(js_ok, key=lambda j: float(j.get("t_rel", 0.0)))
    raw = [np.asarray(j.get("p_three") or [], dtype=np.float32) for j in js_ok]
    if not raw or any(len(p) == 0 for p in raw):
        return None
    smoothed = _smooth_prob_sequence(raw, smooth_radius)
    picked = len(js_ok) - 1
    for i, p in enumerate(smoothed):
        if float(np.max(p)) >= float(tau_conf):
            picked = i
            break
    rep = dict(js_ok[picked])
    p_sel = smoothed[picked]
    pred = int(np.argmax(p_sel))
    rep["pred"] = pred
    rep["p_max"] = float(np.max(p_sel))
    rep["p_three"] = [float(x) for x in p_sel.ravel()]
    rep["rule"] = "e1f_conf_stop"
    rep["e1f_picked_t_rel"] = float(js_ok[picked].get("t_rel", primary_s))
    rep["e1f_tau_conf"] = float(tau_conf)
    rep["e1f_smooth_radius"] = int(smooth_radius)
    return rep


def _mean_slice(probs: Sequence[np.ndarray], lo: int, hi: int) -> np.ndarray:
    """probs[lo:hi] 平均；hi 为开区间。"""
    sl = [np.asarray(probs[j], dtype=np.float32) for j in range(lo, hi)]
    return np.mean(np.stack(sl, axis=0), axis=0).astype(np.float32)


def streaming_conf_stop_S(
    probs: Sequence[np.ndarray],
    *,
    t_rels: Optional[Sequence[float]] = None,
    tau_conf: float = 0.4,
) -> Dict:
    """方案 30 臂 S：等 n+1 到达后，对中心 n 做 (n-1,n,n+1) 平滑 + τ 早停。

    t_dec = 窗 n+1 的到达时刻。边界不足三窗时对可得索引取平均。
    """
    raw = [np.asarray(p, dtype=np.float32) for p in probs]
    k = len(raw)
    if k == 0:
        raise ValueError("empty probs")
    if t_rels is None:
        t_rels = [3.0 + 0.1 * i for i in range(k)]
    picked_center = k - 1
    p_sel = raw[-1]
    t_dec = float(t_rels[-1])
    submitted = False
    for i in range(1, k):
        n = i - 1
        lo = max(0, n - 1)
        hi = n + 2  # exclusive; includes n+1 == i
        p_s = _mean_slice(raw, lo, min(hi, k))
        if float(np.max(p_s)) >= float(tau_conf):
            picked_center = n
            p_sel = p_s
            t_dec = float(t_rels[i])
            submitted = True
            break
    if not submitted:
        if k >= 2:
            n = k - 2
            lo = max(0, n - 1)
            p_sel = _mean_slice(raw, lo, k)
            picked_center = n
            t_dec = float(t_rels[k - 1])
        else:
            p_sel = raw[0]
            picked_center = 0
            t_dec = float(t_rels[0])
    pred = int(np.argmax(p_sel))
    return {
        "pred": pred,
        "p_max": float(np.max(p_sel)),
        "p_three": [float(x) for x in p_sel.ravel()],
        "rule": "streaming_S_conf_stop",
        "center_idx": int(picked_center),
        "t_dec": float(t_dec),
        "tau_conf": float(tau_conf),
        "early": bool(submitted),
    }


def streaming_conf_stop_C(
    probs: Sequence[np.ndarray],
    *,
    t_rels: Optional[Sequence[float]] = None,
    tau_conf: float = 0.4,
    min_windows: int = 3,
) -> Dict:
    """方案 30 臂 C：当前 n 用 (n-2,n-1,n) 因果平滑 + 同款 τ 早停。

    主规则：n+1 >= min_windows（即 n >= min_windows-1）才允许提交。
    t_dec = 当前窗 n 到达时刻。
    """
    raw = [np.asarray(p, dtype=np.float32) for p in probs]
    k = len(raw)
    if k == 0:
        raise ValueError("empty probs")
    if t_rels is None:
        t_rels = [3.0 + 0.1 * i for i in range(k)]
    need = max(1, int(min_windows))
    picked_n = k - 1
    p_sel = _mean_slice(raw, max(0, k - need), k)
    t_dec = float(t_rels[-1])
    submitted = False
    for n in range(k):
        if n + 1 < need:
            continue
        lo = max(0, n - 2)
        p_s = _mean_slice(raw, lo, n + 1)
        if float(np.max(p_s)) >= float(tau_conf):
            picked_n = n
            p_sel = p_s
            t_dec = float(t_rels[n])
            submitted = True
            break
    if not submitted:
        n = k - 1
        lo = max(0, n - 2)
        p_sel = _mean_slice(raw, lo, n + 1)
        picked_n = n
        t_dec = float(t_rels[n])
    pred = int(np.argmax(p_sel))
    return {
        "pred": pred,
        "p_max": float(np.max(p_sel)),
        "p_three": [float(x) for x in p_sel.ravel()],
        "rule": "streaming_C_conf_stop",
        "center_idx": int(picked_n),
        "t_dec": float(t_dec),
        "tau_conf": float(tau_conf),
        "early": bool(submitted),
    }


def majority_vote_from_probs(
    probs: Sequence[np.ndarray],
    *,
    t_rels: Optional[Sequence[float]] = None,
) -> Dict:
    """满试次窗级多数票（臂 W）；平票取该类概率之和最大。"""
    raw = [np.asarray(p, dtype=np.float32) for p in probs]
    if not raw:
        raise ValueError("empty probs")
    if t_rels is None:
        t_rels = [3.0 + 0.1 * i for i in range(len(raw))]
    preds = [int(np.argmax(p)) for p in raw]
    votes: Dict[int, int] = {}
    for pr in preds:
        votes[pr] = votes.get(pr, 0) + 1
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(top) == 1 or top[0][1] > top[1][1]:
        winner = top[0][0]
    else:
        tied = [c for c, n in top if n == top[0][1]]
        sums = {
            c: float(sum(float(raw[i][c]) for i, pr in enumerate(preds) if pr == c))
            for c in tied
        }
        winner = max(tied, key=lambda c: sums[c])
    return {
        "pred": int(winner),
        "p_max": float(max(float(p[winner]) for p in raw)),
        "rule": "majority_vote",
        "vote_counts": {int(k): int(v) for k, v in votes.items()},
        "t_dec": float(t_rels[-1]),
        "early": False,
    }


def causal_smooth_pred_sequence(
    probs: Sequence[np.ndarray],
    *,
    lookback: int = 2,
) -> List[Dict]:
    """对窗概率序列做因果平滑后 argmax。

    窗 n 使用 ``mean(p[n-lookback], …, p[n])``（边界裁剪）；返回每窗
    ``{pred, p_max, p_three}``。
    """
    raw = [np.asarray(p, dtype=np.float32) for p in probs]
    out: List[Dict] = []
    lb = max(0, int(lookback))
    for n in range(len(raw)):
        lo = max(0, n - lb)
        p_s = _mean_slice(raw, lo, n + 1)
        pred = int(np.argmax(p_s))
        out.append(
            {
                "pred": pred,
                "p_max": float(np.max(p_s)),
                "p_three": [float(x) for x in p_s.ravel()],
            }
        )
    return out
