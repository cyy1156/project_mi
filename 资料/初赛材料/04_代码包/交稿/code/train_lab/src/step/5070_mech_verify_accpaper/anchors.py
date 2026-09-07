"""方案 23 · 训练/评估锚点划分（A1_all 尾窗扩充）。"""
from __future__ import annotations

import numpy as np

CANVAS = 1000
HOP_SEC = 0.1
FS = 250.0
T0_TAIL_MAX = 3.9


def eval_indices(all_idx: np.ndarray) -> np.ndarray:
    return np.asarray(all_idx, dtype=np.int64)


def train_indices_future_complete(tr_idx: np.ndarray) -> np.ndarray:
    return np.asarray(tr_idx, dtype=np.int64)


def _to_bct(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x.squeeze(0)
    if x.shape[0] in (500, 600, 750, 875, 1000) and x.shape[1] == 8:
        x = x.T
    return x


def expand_train_tail_windows(
    x_full: np.ndarray,
    y: np.ndarray,
    trial_id: np.ndarray,
    t0_sec: np.ndarray,
    subjects: np.ndarray,
    tr_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A1_all：每 trial 在末锚点后按 hop 右移，future 不足处零填。"""
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    base_n = int(len(x_full))
    groups: dict[int, list[int]] = {}
    for j in tr_idx:
        groups.setdefault(int(trial_id[j]), []).append(int(j))

    extra_x: list[np.ndarray] = []
    extra_y: list[int] = []
    extra_tid: list[int] = []
    extra_t0: list[float] = []
    extra_subj: list[str] = []

    for tid, js in groups.items():
        js_arr = np.asarray(js, dtype=np.int64)
        j_last = int(js_arr[np.argmax(t0_sec[js_arr])])
        xf0 = _to_bct(x_full[j_last])
        t_last = float(t0_sec[j_last])
        subj = str(subjects[j_last])
        lab = int(y[j_last])
        t_new = t_last + HOP_SEC
        while t_new <= T0_TAIL_MAX + 1e-6:
            shift = int(round((t_new - t_last) * FS))
            seg = xf0[..., shift : shift + CANVAS]
            if seg.shape[-1] < CANVAS:
                pad = np.zeros((8, CANVAS - seg.shape[-1]), dtype=np.float32)
                seg = np.concatenate([seg, pad], axis=-1)
            extra_x.append(seg)
            extra_y.append(lab)
            extra_tid.append(int(tid))
            extra_t0.append(float(t_new))
            extra_subj.append(subj)
            t_new += HOP_SEC

    if not extra_x:
        return x_full, y, trial_id, t0_sec, subjects, tr_idx

    stack = np.stack(extra_x, axis=0).astype(np.float32)
    base = np.asarray(x_full, dtype=np.float32)
    if base.ndim == 4 and base.shape[1] == 1:
        base = base[:, 0]
    x_ext = np.concatenate([base, stack], axis=0)
    y_ext = np.concatenate([y, np.asarray(extra_y, dtype=np.int64)])
    tid_ext = np.concatenate([trial_id, np.asarray(extra_tid, dtype=np.int64)])
    t0_ext = np.concatenate([t0_sec, np.asarray(extra_t0, dtype=np.float32)])
    subj_ext = np.concatenate([subjects, np.asarray(extra_subj, dtype=object)])
    extra_idx = np.arange(base_n, base_n + len(extra_x), dtype=np.int64)
    tr_ext = np.concatenate([tr_idx, extra_idx])
    return x_ext, y_ext, tid_ext, t0_ext, subj_ext, tr_ext


def prepare_fold_arrays(
    arm_id: str,
    x_full: np.ndarray,
    y: np.ndarray,
    trial_id: np.ndarray,
    t0_sec: np.ndarray,
    subjects: np.ndarray,
    tr_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if arm_id == "A1_all":
        return expand_train_tail_windows(
            x_full, y, trial_id, t0_sec, subjects, tr_idx
        )
    return x_full, y, trial_id, t0_sec, subjects, train_indices_future_complete(tr_idx)
