"""导出窗级质量权重（A2）：按试次偏侧/对侧 Mu 代理分 → 分位 0.5/1.0/1.5。

说明：为与训练窗对齐，在 openbmi_2s_hop100 的已切窗上估计（非重新读 mat）。
idle 试次默认权重 1.0；MI 试次按 (同侧−对侧) Mu 功率差排序后分位赋权。
Val/Test 训练时不使用权重，但数组与 X 等长以便按 mask 切片。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
for p in (STEP, STEP / "baselines_2s_hop100", STEP / "baselines_single", HERE):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from data_paths import resolve_data  # noqa: E402

from channel_fe import IX  # noqa: E402

OUT = (
    Path(__file__).resolve().parents[3]
    / "out"
    / "_fe_cache"
    / "openbmi_2s_hop100_trial_quality_weights.npy"
)
META = OUT.with_name("openbmi_2s_hop100_trial_quality_meta.npz")


def _mu_power(win_ct: np.ndarray) -> np.ndarray:
    """(T,C) or (C,T) → Mu power per channel; expect (C,T) from pack layout (8,T)."""
    if win_ct.shape[0] < win_ct.shape[-1] and win_ct.shape[0] <= 16:
        x = win_ct.T  # (T,C)
    else:
        x = win_ct
    t = x.shape[0]
    freqs = np.fft.rfftfreq(t, d=1.0 / 250.0)
    spec = np.abs(np.fft.rfft(x.astype(np.float64), axis=0)) ** 2
    m = (freqs >= 8.0) & (freqs < 13.0)
    if not np.any(m):
        return np.full(x.shape[1], 1e-12)
    return spec[m].mean(axis=0) + 1e-12


def main() -> None:
    data_dir, prefix = resolve_data("openbmi_2s_hop100")
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y3 = np.load(data_dir / f"{prefix}_y_three.npy")
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    n = len(X)
    assert len(y3) == n == len(trial_ids)

    # unique trials preserving first-seen order
    _, first_idx = np.unique(trial_ids, return_index=True)
    order = np.argsort(first_idx)
    uniq = trial_ids[first_idx[order]]

    trial_score = {}
    for tid in uniq:
        m = trial_ids == tid
        labs = y3[m]
        lab = int(np.bincount(labs.astype(np.int64)).argmax())
        if lab == 0:
            trial_score[int(tid)] = None  # idle → later fill 1.0
            continue
        # sample up to 32 windows for speed
        idxs = np.flatnonzero(m)
        if len(idxs) > 32:
            idxs = idxs[:: max(1, len(idxs) // 32)]
        mus = []
        for i in idxs:
            w = np.array(X[i], dtype=np.float32)
            if w.ndim == 3:
                w = w[0]
            mus.append(_mu_power(w))
        mu = np.mean(np.stack(mus, 0), axis=0)
        c3, c4 = float(mu[IX["C3"]]), float(mu[IX["C4"]])
        # left→期望 C4 低；right→期望 C3 低
        if lab == 1:
            score = (c3 - c4) / (c3 + c4 + 1e-12)
        else:
            score = (c4 - c3) / (c3 + c4 + 1e-12)
        trial_score[int(tid)] = float(score)

    mi_scores = np.array([v for v in trial_score.values() if v is not None], dtype=float)
    q1, q2 = np.quantile(mi_scores, [1 / 3, 2 / 3]) if len(mi_scores) else (0.0, 0.0)

    def w_of(s: float | None) -> float:
        if s is None:
            return 1.0
        if s >= q2:
            return 1.5
        if s <= q1:
            return 0.5
        return 1.0

    weights = np.empty(n, dtype=np.float32)
    for tid in uniq:
        m = trial_ids == tid
        weights[m] = w_of(trial_score[int(tid)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, weights)
    np.savez(
        META,
        q1=q1,
        q2=q2,
        n_trials=len(uniq),
        n_mi=int(np.sum([v is not None for v in trial_score.values()])),
        mean_w=float(weights.mean()),
    )
    print(f"[done] {OUT} mean_w={weights.mean():.3f} q1={q1:.4f} q2={q2:.4f}")
    print(f"  meta → {META}")


if __name__ == "__main__":
    main()
