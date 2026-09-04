# -*- coding: utf-8 -*-
"""Exp40 H1/H2：R-B8 OOF 上 leave-fold / pooled 类边际校正。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_STEP = Path(__file__).resolve().parent
from exp40_config import (  # noqa: E402
    FOLD_OK_MIN,
    RB8_ANCHOR,
    exp39_out,
    exp40_out,
)


def _apply_bias(probs: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    logits = np.log(p) + b.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _fit_bias(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.astype(int)

    def nll(b: np.ndarray) -> float:
        q = _apply_bias(probs, b)
        q = np.clip(q, 1e-12, 1.0)
        return float(-np.log(q[np.arange(len(y)), y]).mean())

    res = minimize(nll, np.zeros(3), method="L-BFGS-B")
    return np.asarray(res.x, dtype=np.float64)


def _accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y.astype(int)).mean())


def _split_folds(prob: np.ndarray, y: np.ndarray, n_folds: int = 6) -> tuple[list, list]:
    n = len(y)
    assert n % n_folds == 0, f"n={n} not divisible by {n_folds}"
    m = n // n_folds
    probs_l = [prob[i * m : (i + 1) * m] for i in range(n_folds)]
    y_l = [y[i * m : (i + 1) * m] for i in range(n_folds)]
    return probs_l, y_l


def fold_ok(arm_accs: list[float], base_accs: list[float]) -> dict:
    n_ge = int(sum(1 for a, b in zip(arm_accs, base_accs) if a + 1e-12 >= b))
    return {"n_folds_ge": n_ge, "fold_ok": bool(n_ge >= FOLD_OK_MIN)}


def run_margin(rb8_prob: np.ndarray, rb8_y: np.ndarray, base_fold_accs: list[float]) -> dict:
    probs_l, y_l = _split_folds(rb8_prob, rb8_y)
    n_folds = len(probs_l)
    base = list(base_fold_accs)

    nested_accs, nested_folds, nested_probs = [], [], []
    for i in range(n_folds):
        p_tr = np.concatenate([probs_l[j] for j in range(n_folds) if j != i], axis=0)
        y_tr = np.concatenate([y_l[j] for j in range(n_folds) if j != i], axis=0)
        b = _fit_bias(p_tr, y_tr)
        p_te = _apply_bias(probs_l[i], b)
        acc = _accuracy(p_te, y_l[i])
        nested_accs.append(acc)
        nested_probs.append(p_te)
        nested_folds.append({"fold": i, "bias": b.tolist(), "acc": acc})
        print(f"  MC-B8 fold{i} acc={acc:.4f} (base={base[i]:.4f})", flush=True)

    p_all = np.concatenate(probs_l, axis=0)
    y_all = np.concatenate(y_l, axis=0)
    b_pool = _fit_bias(p_all, y_all)
    pooled_fold_accs, pooled_probs = [], []
    for i in range(n_folds):
        p_i = _apply_bias(probs_l[i], b_pool)
        pooled_probs.append(p_i)
        pooled_fold_accs.append(_accuracy(p_i, y_l[i]))

    fo = fold_ok(nested_accs, base)
    mean = float(np.mean(nested_accs))
    std = float(np.std(nested_accs, ddof=1)) if n_folds > 1 else 0.0
    delta = mean - float(np.mean(base))
    enter = bool(mean + 1e-12 >= RB8_ANCHOR and fo["fold_ok"])

    return {
        "arm_id": "MC-B8",
        "val_acc_mean": mean,
        "val_acc_std": std,
        "fold_accs": nested_accs,
        "folds": nested_folds,
        "delta_vs_rb8": delta,
        **fo,
        "enter_candidate": enter,
        "pooled": {
            "arm_id": "MC-pool-B8",
            "bias": b_pool.tolist(),
            "fold_accs": pooled_fold_accs,
            "val_acc_mean": float(np.mean(pooled_fold_accs)),
            "val_acc_std": float(np.std(pooled_fold_accs, ddof=1)) if n_folds > 1 else 0.0,
            "pooled_acc_900": _accuracy(_apply_bias(p_all, b_pool), y_all),
            "note": "in-sample-b diagnostic; not ranking key",
        },
        "_nested_probs": nested_probs,
        "_pooled_probs": pooled_probs,
        "_y_folds": y_l,
    }


def main() -> int:
    preds = exp39_out() / "preds"
    ranking = json.loads((exp39_out() / "replay" / "ranking_latest.json").read_text(encoding="utf-8"))
    base_accs = ranking["arms"]["R-B8"]["fold_accs"]
    prob = np.load(preds / "oof_R-B8_prob.npy")
    y = np.load(preds / "oof_R-B8_y.npy")
    print("=== H1/H2 MC-B8 ===", flush=True)
    arm = run_margin(prob, y, base_accs)
    out = exp40_out() / "replay"
    out.mkdir(parents=True, exist_ok=True)
    pub = {k: v for k, v in arm.items() if not k.startswith("_")}
    (out / "margin_latest.json").write_text(
        json.dumps(pub, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(out / "mc_b8_nested_prob.npy", np.concatenate(arm["_nested_probs"], axis=0))
    np.save(out / "mc_pool_b8_prob.npy", np.concatenate(arm["_pooled_probs"], axis=0))
    print(
        f"MC-B8 nested={arm['val_acc_mean']:.4f} Δ={arm['delta_vs_rb8']*100:+.2f}pp "
        f"fold_ok={arm['fold_ok']} enter={arm['enter_candidate']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
