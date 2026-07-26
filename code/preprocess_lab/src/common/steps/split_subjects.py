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


def make_subject_groups(
   unique_subjects: list[str],
   n_folds: int =5,
   seed: int = 42,
) ->list[list[str]]:
    """
        作用：把全部被试打乱后，尽量均分成 n_folds 组。

        例子（10 人, n_folds=5）→
          [["A01","A02"], ["A03","A04"], ... 共 5 组]
        之后：第 k 折用第 k 组当 test。

        注意：seed 固定后，分组结果可复现；不要用 5 个 seed 冒充五折。
    """
    rng = np.random.default_rng(seed)
    subs =np.array(sorted(unique_subjects))
    rng.shuffle(subs)
    groups = [list(g) for g in np.array_split(subs, n_folds)]
    return groups
def split_train_val_subjects(
    remain_subjects: list[str],
    val_ratio: float =0.2,
    seed: int =42,
    fold_id:int =0,
)->tuple[list[str], list[str]]:
    """
        作用：从「本折非测试被试」里，按人抽出约 val_ratio 做验证集。

        例子：remain=80 人, val_ratio=0.2 → val≈16 人, train≈64 人。

        seed + fold_id：每折可复现，且折与折之间抽到的 val 人不完全相同。
    """
    rng = np.random.default_rng(seed+fold_id)
    remain=np.array(sorted(remain_subjects))
    rng.shuffle(remain)

    n_val=0
    if len(remain)>1:
        n_val = max(1,int(round(len(remain)*val_ratio)))
        n_val = min(n_val, len(remain) - 1)  # 至少留 1 人给 train

    val_subjects =remain[:n_val].tolist()
    train_subjects = remain[n_val:].tolist()
    return train_subjects,val_subjects


def iter_subject_kfold(
    subjects_per_trial:np.ndarray,
    n_folds: int =5,
    val_ratio: float =0.2,
    seed: int = 42,
):
    """
       作用：生成每一折的划分结果（按人独立）。

       参数 subjects_per_trial：
         形状 (N,)，与 X 第 0 维等长。
         例如第 i 条试次来自被试 "A03"，则 subjects_per_trial[i] == "A03"。
         对应文件：preprocess_lab/out/bci2a/bci2a_subjects.npy

       每一折 yield 一个 dict：
         fold            : 0..4
         train_subjects  : 本折训练用的人名列表
         val_subjects    : 本折验证用的人名列表
         test_subjects   : 本折测试用的人名列表
         masks           : {"train"|"val"|"test" -> 长度为 N 的 bool 数组}
                           True 表示该试次属于该集合

       训练时用法：
         X_tr = X[masks["train"]]
         y_tr = y[masks["train"]]
         ...
    """
    unique=sorted(set(subjects_per_trial.tolist()))
    groups=make_subject_groups(unique, n_folds=n_folds, seed=seed)

    for fold_id,test_subjects in enumerate(groups):
        remain =[s for i,g in enumerate(groups) if i != fold_id for s in g]
        train_subjects,val_subjects = split_train_val_subjects(
            remain,val_ratio=val_ratio,seed=seed,fold_id=fold_id
        )
        subj = np.asarray(subjects_per_trial)
        masks={
            "train":np.isin(subj, train_subjects),
            "val":np.isin(subj, val_subjects),
            "test":np.isin(subj, test_subjects),
        }

        assert not set(train_subjects)&set(val_subjects)
        assert not set(test_subjects)&set(val_subjects)
        assert not set(test_subjects)&set(train_subjects)

        yield {
            "fold": fold_id,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": list(test_subjects),
            "masks": masks,
        }

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


def _test_iter_subject_kfold() -> None:
    # 10 人各 5 试次 → 检查五折互斥与覆盖
    subjects = np.array([f"S{i:02d}" for i in range(10) for _ in range(5)])
    seen_test = []
    for info in iter_subject_kfold(subjects, n_folds=5, val_ratio=0.2, seed=42):
        tr, va, te = set(info["train_subjects"]), set(info["val_subjects"]), set(info["test_subjects"])
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert info["masks"]["train"].sum() + info["masks"]["val"].sum() + info["masks"]["test"].sum() == len(subjects)
        seen_test.extend(info["test_subjects"])
    assert sorted(seen_test) == [f"S{i:02d}" for i in range(10)]
    print("iter_subject_kfold OK")


def main() -> None:
    _test_split_all_trials()
    _test_split_by_subject()
    _test_iter_subject_kfold()


if __name__ == "__main__":
    main()
