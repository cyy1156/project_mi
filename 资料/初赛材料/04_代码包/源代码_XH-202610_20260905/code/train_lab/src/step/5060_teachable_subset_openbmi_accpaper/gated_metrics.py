"""试次级 Acc_paper + 门控：低质量窗不入投票；空窗试次 abstain。"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_STEP = Path(__file__).resolve().parent.parent
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

from metrics import binary_task_metrics, jsonify_metrics, three_class_metrics

TIE_SENTINEL = -1


def _majority_label(preds: np.ndarray) -> int:
    cnt = Counter(int(p) for p in preds.tolist())
    top = cnt.most_common()
    if not top:
        return TIE_SENTINEL
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return TIE_SENTINEL
    return int(top[0][0])


def aggregate_windows_to_trials_gated(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    keep: np.ndarray,
    *,
    n_classes: int,
) -> dict:
    """对全部试次聚合；``keep`` 为与窗对齐的 bool。

    - 试次内无 keep 窗 → abstain（不计 Acc_paper 分母）
    - 另报 abstain_as_wrong_acc（abstain 计错）
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    subjects = np.asarray([str(s) for s in subjects])
    trial_ids = np.asarray(trial_ids).astype(np.int64).reshape(-1)
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    assert len(y_true) == len(y_pred) == len(subjects) == len(trial_ids) == len(keep)

    keys = list(zip(subjects.tolist(), trial_ids.tolist()))
    order: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)

    trial_y: list[int] = []
    trial_pred_maj: list[int] = []
    trial_paper_ok: list[bool] = []
    trial_abstain: list[bool] = []
    trial_n_win: list[int] = []
    trial_n_keep: list[int] = []
    trial_correct_rate: list[float] = []
    trial_subjects: list[str] = []
    trial_ids_out: list[int] = []

    for subj, tid in order:
        mask = (subjects == subj) & (trial_ids == tid)
        yt_all = y_true[mask]
        uniq = np.unique(yt_all)
        if len(uniq) != 1:
            raise ValueError(
                f"trial ({subj}, {tid}) 标签不恒定: {uniq.tolist()} n={len(yt_all)}"
            )
        y = int(uniq[0])
        kmask = keep[mask]
        n_all = int(mask.sum())
        n_keep = int(kmask.sum())
        trial_subjects.append(subj)
        trial_ids_out.append(int(tid))
        trial_y.append(y)
        trial_n_win.append(n_all)
        trial_n_keep.append(n_keep)

        if n_keep == 0:
            trial_abstain.append(True)
            trial_paper_ok.append(False)
            trial_pred_maj.append(TIE_SENTINEL)
            trial_correct_rate.append(float("nan"))
            continue

        yt = yt_all[kmask]
        yp = y_pred[mask][kmask]
        rate = float(np.mean(yp == y))
        paper_ok = rate > 0.5
        maj = _majority_label(yp)
        if maj == TIE_SENTINEL:
            maj_for_metric = (y + 1) % n_classes
        else:
            maj_for_metric = maj

        trial_abstain.append(False)
        trial_paper_ok.append(paper_ok)
        trial_pred_maj.append(maj_for_metric)
        trial_correct_rate.append(rate)

    yt_arr = np.asarray(trial_y, dtype=np.int64)
    abstain_arr = np.asarray(trial_abstain, dtype=bool)
    paper_ok_arr = np.asarray(trial_paper_ok, dtype=bool)
    n_all_trials = int(len(yt_arr))
    n_abstain = int(abstain_arr.sum())
    n_scored = n_all_trials - n_abstain
    scored_ok = paper_ok_arr[~abstain_arr]
    acc_paper = float(scored_ok.mean()) if n_scored else float("nan")
    # abstain 计错：正确数 / 全部试次
    n_correct = int(scored_ok.sum()) if n_scored else 0
    abstain_as_wrong_acc = float(n_correct / n_all_trials) if n_all_trials else float("nan")
    abstain_rate = float(n_abstain / n_all_trials) if n_all_trials else 0.0

    # 众数指标仅在非 abstain 上
    yp_maj = np.asarray(trial_pred_maj, dtype=np.int64)
    scored_yt = yt_arr[~abstain_arr]
    scored_yp = yp_maj[~abstain_arr]
    if n_scored == 0:
        metrics = {
            "n_trials_all": n_all_trials,
            "n_trials_scored": 0,
            "n_trials_abstain": n_abstain,
            "n_windows": int(len(y_true)),
            "n_windows_kept": int(keep.sum()),
            "acc_paper": float("nan"),
            "abstain_rate": abstain_rate,
            "abstain_as_wrong_acc": abstain_as_wrong_acc,
            "acc_majority": float("nan"),
        }
    elif n_classes == 2:
        m = binary_task_metrics(scored_yt, scored_yp)
        metrics = {
            "n_trials_all": n_all_trials,
            "n_trials_scored": n_scored,
            "n_trials_abstain": n_abstain,
            "n_windows": int(len(y_true)),
            "n_windows_kept": int(keep.sum()),
            "acc_paper": acc_paper,
            "abstain_rate": abstain_rate,
            "abstain_as_wrong_acc": abstain_as_wrong_acc,
            "acc_majority": float(np.mean(scored_yp == scored_yt)),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1": float(m["f1"]),
            "recall": float(m["recall"]),
            "specificity": float(m["specificity"]),
        }
    elif n_classes == 3:
        m = three_class_metrics(scored_yt, scored_yp)
        metrics = {
            "n_trials_all": n_all_trials,
            "n_trials_scored": n_scored,
            "n_trials_abstain": n_abstain,
            "n_windows": int(len(y_true)),
            "n_windows_kept": int(keep.sum()),
            "acc_paper": acc_paper,
            "abstain_rate": abstain_rate,
            "abstain_as_wrong_acc": abstain_as_wrong_acc,
            "acc_majority": float(np.mean(scored_yp == scored_yt)),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1_macro": float(m["f1_macro"]),
            "recall_idle": float(m["recall_idle"]),
            "recall_left": float(m["recall_left"]),
            "recall_right": float(m["recall_right"]),
            "cm": m["cm"],
        }
    else:
        raise ValueError(f"n_classes={n_classes}")

    return {
        "metrics": jsonify_metrics(metrics),
        "trial_y": yt_arr,
        "trial_abstain": abstain_arr,
        "trial_paper_ok": paper_ok_arr,
        "trial_n_windows": np.asarray(trial_n_win, dtype=np.int64),
        "trial_n_keep": np.asarray(trial_n_keep, dtype=np.int64),
        "trial_subjects": np.asarray(trial_subjects, dtype=object),
        "trial_ids": np.asarray(trial_ids_out, dtype=np.int64),
    }


def build_g3_keep_mask(
    X,
    y_three: np.ndarray,
    trial_ids: np.ndarray,
    *,
    top_p: float = 0.5,
    fs: float = 250.0,
) -> np.ndarray:
    """试次内相对门控：REST 全留；MI 按窗级偏侧代理 top-p%。"""
    y_three = np.asarray(y_three).astype(int).reshape(-1)
    trial_ids = np.asarray(trial_ids).astype(np.int64).reshape(-1)
    n = len(y_three)
    keep = np.zeros(n, dtype=bool)
    # C3=1, C4=2 in hop100 channel order
    c3, c4 = 1, 2

    def mu_pow(win) -> tuple[float, float]:
        x = np.asarray(win, dtype=np.float64)
        if x.ndim == 3:
            x = x[0]
        # (C,T)
        if x.shape[0] < x.shape[-1]:
            t = x.shape[-1]
            freqs = np.fft.rfftfreq(t, d=1.0 / fs)
            spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
            m = (freqs >= 8.0) & (freqs < 13.0)
            band = spec[:, m].mean(axis=-1) + 1e-12
            return float(band[c3]), float(band[c4])
        t = x.shape[0]
        freqs = np.fft.rfftfreq(t, d=1.0 / fs)
        spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
        m = (freqs >= 8.0) & (freqs < 13.0)
        band = spec[m].mean(axis=0) + 1e-12
        return float(band[c3]), float(band[c4])

    _, first = np.unique(trial_ids, return_index=True)
    for tid in trial_ids[np.sort(first)]:
        idxs = np.flatnonzero(trial_ids == tid)
        lab = int(np.bincount(y_three[idxs].astype(np.int64)).argmax())
        if lab == 0:
            keep[idxs] = True
            continue
        scores = []
        for i in idxs:
            p3, p4 = mu_pow(X[i])
            if lab == 1:  # left → want C4 low
                s = (p3 - p4) / (p3 + p4 + 1e-12)
            else:
                s = (p4 - p3) / (p3 + p4 + 1e-12)
            scores.append(s)
        scores = np.asarray(scores, dtype=float)
        k = max(1, int(np.ceil(len(scores) * float(top_p))))
        # top-k by score
        order = np.argsort(-scores)
        chosen = idxs[order[:k]]
        keep[chosen] = True
    return keep
