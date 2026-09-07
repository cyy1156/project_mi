"""6-fold LOSO：Val=1 被试，Train=其余 5；盲测 S07–S08 不进折内。"""

from __future__ import annotations

from typing import Iterator

import numpy as np

CANONICAL_TRAIN_SUBJECTS = [
    "challenge:S01",
    "challenge:S02",
    "challenge:S03",
    "challenge:S04",
    "challenge:S05",
    "challenge:S06",
]


def unique_subjects(subjects: np.ndarray) -> list[str]:
    seen: list[str] = []
    for s in subjects.tolist():
        ss = str(s)
        if ss not in seen:
            seen.append(ss)
    return seen


def iter_loso6(
    subjects: np.ndarray,
    *,
    fold_subjects: list[str] | None = None,
) -> Iterator[dict]:
    """
    Yields dict:
      fold, val_subjects, train_subjects,
      masks: {train, val} boolean arrays (len=N)
    """
    subj = np.asarray([str(s) for s in subjects])
    order = list(fold_subjects) if fold_subjects else unique_subjects(subj)
    # 尽量按 S01..S06 排序
    preferred = [s for s in CANONICAL_TRAIN_SUBJECTS if s in order]
    rest = [s for s in order if s not in preferred]
    order = preferred + rest
    if len(order) < 2:
        raise RuntimeError(f"LOSO 需要 ≥2 被试，收到 {order}")

    for fold, val_s in enumerate(order):
        train_s = [s for s in order if s != val_s]
        m_val = subj == val_s
        m_tr = np.isin(subj, train_s)
        if not m_val.any() or not m_tr.any():
            raise RuntimeError(
                f"fold{fold} 空集: val={val_s} n_val={int(m_val.sum())} "
                f"n_tr={int(m_tr.sum())}"
            )
        yield {
            "fold": fold,
            "val_subjects": [val_s],
            "train_subjects": train_s,
            "masks": {"train": m_tr, "val": m_val},
        }
