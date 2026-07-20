"""Step 12: trial split helpers.

Primary (offline baseline): split_all_trials — pool all subjects, 8:2 train/val.
Later (cross-subject): split_by_subject — whole subjects to train/val/test.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_all_trials(
    X: np.ndarray,
    y_task: np.ndarray,
    y_three: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
    subjects: list[str] | np.ndarray | None = None,
) -> dict:
    """
    Split all pooled trials by ratio (default train:val = 8:2).

    Returns dict of "train"/"val"/"test" ->
      (X, y_task, y_three) or (X, y_task, y_three, subjects).
    """
    assert len(X) == len(y_task) == len(y_three)
    subj_arr: np.ndarray | None = None
    if subjects is not None:
        subj_arr = np.asarray(subjects)
        assert len(subj_arr) == len(X)
    assert 0.0 < val_ratio < 1.0
    assert 0.0 <= test_ratio < 1.0
    assert val_ratio + test_ratio < 1.0

    idx = np.arange(len(X))
    out: dict = {}

    def _pack(ii: np.ndarray):
        if subj_arr is None:
            return X[ii], y_task[ii], y_three[ii]
        return X[ii], y_task[ii], y_three[ii], subj_arr[ii]

    if test_ratio > 0:
        idx_tv, idx_test = train_test_split(
            idx,
            test_size=test_ratio,
            random_state=seed,
            stratify=y_three,
        )
        out["test"] = _pack(idx_test)
        val_ratio_adj = val_ratio / (1.0 - test_ratio)
        idx_train, idx_val = train_test_split(
            idx_tv,
            test_size=val_ratio_adj,
            random_state=seed,
            stratify=y_three[idx_tv],
        )
    else:
        idx_train, idx_val = train_test_split(
            idx,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_three,
        )

    out["train"] = _pack(idx_train)
    out["val"] = _pack(idx_val)
    return out


def split_by_subject(
    X: np.ndarray,
    y_task: np.ndarray,
    y_three: np.ndarray,
    subjects: list[str],
    test_subjects: set[str],
    val_subjects: set[str],
) -> dict:
    """
    Cross-subject split: each subject goes entirely into one split.
    Returns {"train"|"val"|"test": (X, y_task, y_three)}.
    """
    assert len(X) == len(y_task) == len(y_three) == len(subjects)
    assert test_subjects.isdisjoint(val_subjects), "val 与 test 被试不能重叠"

    subj = np.asarray(subjects)
    holdout = set(test_subjects) | set(val_subjects)
    masks = {
        "train": ~np.isin(subj, list(holdout)),
        "val": np.isin(subj, list(val_subjects)),
        "test": np.isin(subj, list(test_subjects)),
    }
    return {k: (X[m], y_task[m], y_three[m]) for k, m in masks.items()}


def _test_split_all_trials() -> None:
    X = np.zeros((100, 1, 8, 1000), np.float32)
    y_task = np.array([1] * 70 + [0] * 30)
    y_three = np.array([1] * 35 + [2] * 35 + [0] * 30)
    subjects = ["A01"] * 40 + ["A02"] * 30 + ["A03"] * 30

    parts = split_all_trials(
        X, y_task, y_three, val_ratio=0.2, seed=42, subjects=subjects
    )
    n_train = parts["train"][0].shape[0]
    n_val = parts["val"][0].shape[0]
    print("train/val:", n_train, n_val)
    assert n_train + n_val == 100
    assert "test" not in parts

    for name, pack in parts.items():
        Xi, yt, y3, sid = pack
        assert len(Xi) == len(yt) == len(y3) == len(sid)
        print(name, "y_three:", np.bincount(y3, minlength=3), "subjects:", sorted(set(sid)))

    print("split_all_trials OK")


def _test_split_by_subject() -> None:
    X = np.zeros((10, 1, 8, 1000), np.float32)
    y_task = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    y_three = np.array([1, 2, 1, 2, 1, 2, 1, 0, 0, 0])
    subjects = ["A01"] * 4 + ["A02"] * 3 + ["A03"] * 3

    parts = split_by_subject(
        X,
        y_task,
        y_three,
        subjects,
        test_subjects={"A03"},
        val_subjects={"A02"},
    )
    assert parts["train"][0].shape[0] == 4
    assert parts["val"][0].shape[0] == 3
    assert parts["test"][0].shape[0] == 3
    print("split_by_subject OK")


def main() -> None:
    _test_split_all_trials()
    _test_split_by_subject()


if __name__ == "__main__":
    main()
