# -*- coding: utf-8 -*-
"""Exp37 辅证：McNemar + 被试级 cluster bootstrap；Wilcoxon / t。"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, ttest_rel, wilcoxon


def wilcoxon_p(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float | None:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if len(d) < 1:
        return None
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return None


def paired_t_p(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if len(aa) < 2:
        return None
    try:
        return float(ttest_rel(aa, bb).pvalue)
    except ValueError:
        return None


def mcnemar_exact(y_hat_a: np.ndarray, y_hat_b: np.ndarray, y: np.ndarray) -> dict:
    """A vs B：主报精确二项 McNemar（双侧）。b=A对B错, c=A错B对。"""
    y = np.asarray(y, dtype=np.int64)
    ya = np.asarray(y_hat_a, dtype=np.int64)
    yb = np.asarray(y_hat_b, dtype=np.int64)
    a_ok = ya == y
    b_ok = yb == y
    b = int(np.sum(a_ok & ~b_ok))  # A correct, B wrong
    c = int(np.sum(~a_ok & b_ok))  # A wrong, B correct
    n = b + c
    if n == 0:
        p = 1.0
    else:
        # H0: P(b)=P(c)；精确二项对 min(b,c) under n, p=0.5，双侧
        p = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return {
        "b_a_ok_b_wrong": b,
        "c_a_wrong_b_ok": c,
        "n_discordant": n,
        "p_exact": p,
    }


def cluster_bootstrap_delta(
    correct_arm: np.ndarray,
    correct_base: np.ndarray,
    subject_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """
    被试级有放回重抽：Δ = mean(correct_arm) - mean(correct_base)。
    subject_ids 与试次对齐（如 challenge:S01）。
    """
    correct_arm = np.asarray(correct_arm, dtype=np.float64)
    correct_base = np.asarray(correct_base, dtype=np.float64)
    subject_ids = np.asarray(subject_ids)
    uniq = np.array(sorted(set(subject_ids.tolist())))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(subject_ids, draw)
        # 有放回：按抽到的被试次数加权（重复被试则重复其试次块）
        parts_a, parts_b = [], []
        for s in draw:
            m = subject_ids == s
            parts_a.append(correct_arm[m])
            parts_b.append(correct_base[m])
        ca = np.concatenate(parts_a)
        cb = np.concatenate(parts_b)
        deltas[i] = float(ca.mean() - cb.mean())
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    point = float(correct_arm.mean() - correct_base.mean())
    return {
        "delta": point,
        "ci95": [float(lo), float(hi)],
        "n_boot": int(n_boot),
        "n_subjects": int(len(uniq)),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "suspect_single_subject": _suspect_single(correct_arm, correct_base, subject_ids),
    }


def _suspect_single(
    correct_arm: np.ndarray,
    correct_base: np.ndarray,
    subject_ids: np.ndarray,
) -> dict:
    """若某一被试贡献的 Δ 份额 >50% 总 Δ（当总 Δ>0），标风险。"""
    uniq = sorted(set(subject_ids.tolist()))
    total = float(correct_arm.mean() - correct_base.mean())
    per = {}
    for s in uniq:
        m = subject_ids == s
        # 用该被试内 Δ * 该被试试次占比 近似贡献
        w = float(m.mean())
        d = float(correct_arm[m].mean() - correct_base[m].mean())
        per[str(s)] = {"subj_delta": d, "weight": w, "contrib_approx": d * w}
    if abs(total) < 1e-12:
        return {"flag": False, "reason": "total_delta≈0", "per_subject": per}
    # 最大正贡献占比
    contribs = {k: max(0.0, v["contrib_approx"]) for k, v in per.items()}
    smax = max(contribs, key=contribs.get)
    share = contribs[smax] / max(sum(contribs.values()), 1e-12)
    return {
        "flag": bool(share >= 0.50 and total > 0),
        "top_subject": smax,
        "positive_share": float(share),
        "per_subject": per,
    }
