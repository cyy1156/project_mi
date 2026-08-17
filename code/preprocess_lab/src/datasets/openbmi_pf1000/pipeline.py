"""OpenBMI · past+cur+future 1000pt 预处理臂（新路径，不改旧 openbmi pipeline）。

协议：资料/.../数据切片与边界过滤说明.md
输出：(N,1,8,1000) X_full / X_mask；v1 仅 left/right（无 Rest）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import resample

from src.common.eeg_types import ContinuousEEG
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.resample_zscore import to_model_tensor, trial_zscore
from src.common.steps.select_channels import select_channels
from src.datasets.bci2a.labels import filter_left_right_events
from src.datasets.openbmi.load_mat import load_openbmi_mat

PROTOCOL = "openbmi_2s_hop100_pf1000"
FS_OUT = 250.0
PAST = 100
CUR = 500
FUT = 400
N_TIMES = PAST + CUR + FUT  # 1000
HOP = 25  # 0.1 s @ 250 Hz
MI_SEC = 4.0
POST_SEC = 1.6
SEG_SEC = MI_SEC + POST_SEC  # 5.6
BASELINE_SEC = 0.5
MI_END_PTS = int(round(MI_SEC * FS_OUT))  # 1000


def _extract_seg_from_cue(
    x: np.ndarray,
    cue: int,
    fs: float,
) -> np.ndarray | None:
    """Cue 前 0.5s 基线 + Cue 后 5.6s；基线校正后丢掉基线 → (T_raw, 8) 从 cue 起。"""
    n_base = int(round(BASELINE_SEC * fs))
    n_seg = int(round(SEG_SEC * fs))
    t0 = int(cue) - n_base
    t1 = int(cue) + n_seg
    if t0 < 0 or t1 > x.shape[0]:
        return None
    chunk = x[t0:t1].astype(np.float64, copy=False)
    base = chunk[:n_base].mean(axis=0, keepdims=True)
    seg = chunk[n_base:] - base
    if seg.shape[0] != n_seg:
        return None
    return seg


def _resample_seg(seg_tc: np.ndarray, fs_in: float) -> np.ndarray:
    """(T_in, 8) → (1400, 8) @ 250 Hz。"""
    n_out = int(round(SEG_SEC * FS_OUT))
    if abs(fs_in - FS_OUT) < 1e-6 and seg_tc.shape[0] == n_out:
        return seg_tc.astype(np.float32)
    y = resample(seg_tc, n_out, axis=0)
    return np.asarray(y, dtype=np.float32)


def windows_from_seg250(
    seg250: np.ndarray,
    *,
    lab_task: int,
    lab_three: int,
    zscore: bool = True,
) -> list[tuple[np.ndarray, int, int, float]]:
    """
    seg250: (T>=1400, 8) 从 cue 起。
    返回 list of (x_full_T8 (1000,8), y_task, y_three, t0_sec)。
    """
    if seg250.shape[0] < int(round(SEG_SEC * FS_OUT)):
        return []
    out: list[tuple[np.ndarray, int, int, float]] = []
    # i0: 当前窗起点；合法 i0 ∈ [PAST, MI_END-CUR] step HOP
    for i0 in range(0, MI_END_PTS - CUR + 1, HOP):
        if i0 < PAST:
            continue
        if i0 + CUR > MI_END_PTS:
            continue
        if i0 + CUR + FUT > seg250.shape[0]:
            continue
        past = seg250[i0 - PAST : i0]
        cur = seg250[i0 : i0 + CUR]
        future = seg250[i0 + CUR : i0 + CUR + FUT]
        if past.shape[0] != PAST or cur.shape[0] != CUR or future.shape[0] != FUT:
            continue
        x_full = np.concatenate([past, cur, future], axis=0)  # (1000, 8)
        if zscore:
            x_full = trial_zscore(x_full)
        out.append((x_full.astype(np.float32), int(lab_task), int(lab_three), float(i0) / FS_OUT))
    return out


def preprocess_run_pf1000(
    eeg: ContinuousEEG,
    *,
    zscore: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """
    单段连续流 → X_full, X_mask, y_task, y_three, trial_id(段内), t0_sec, stats。
    X_*: (N,1,8,1000)
    """
    x = select_channels(eeg.x, eeg.ch_names)
    x = car_reference(x)
    x = notch_and_bandpass(x, eeg.fs)

    kept = filter_left_right_events(eeg.events, eeg.artifacts)
    xs_full: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []
    trial_ids: list[int] = []
    t0s: list[float] = []
    tid = 0
    n_drop_trial = 0
    n_cand_anchor = 0
    n_keep_win = 0

    for cue, lab_task, lab_three, _ in kept:
        seg = _extract_seg_from_cue(x, int(cue), eeg.fs)
        if seg is None:
            n_drop_trial += 1
            continue
        seg250 = _resample_seg(seg, eeg.fs)
        # 候选锚点数（含将被裁掉的缺 past）
        n_cand_anchor += len(range(0, MI_END_PTS - CUR + 1, HOP))
        wins = windows_from_seg250(
            seg250, lab_task=int(lab_task), lab_three=int(lab_three), zscore=zscore
        )
        if not wins:
            n_drop_trial += 1
            continue
        for x_full, yt, y3, t0 in wins:
            xs_full.append(x_full)
            y_task.append(yt)
            y_three.append(y3)
            trial_ids.append(tid)
            t0s.append(t0)
            n_keep_win += 1
        tid += 1

    stats = {
        "n_cues": int(len(kept)),
        "n_trials_kept": int(tid),
        "n_trials_dropped": int(n_drop_trial),
        "n_cand_anchors": int(n_cand_anchor),
        "n_windows": int(n_keep_win),
        "protocol": PROTOCOL,
        "zscore": bool(zscore),
        "no_rest": True,
    }

    if not xs_full:
        empty = np.zeros((0, 1, 8, N_TIMES), np.float32)
        z = np.zeros((0,), np.int64)
        return empty, empty.copy(), z, z.copy(), z.copy(), np.zeros((0,), np.float32), stats

    X_full = to_model_tensor(xs_full)  # (N,1,8,1000)
    X_mask = X_full.copy()
    X_mask[..., -FUT:] = 0.0
    return (
        X_full,
        X_mask,
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
        np.asarray(t0s, dtype=np.float32),
        stats,
    )


def preprocess_file_pf1000(
    mat_path: Path | str,
    *,
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """单 mat → X_full, X_mask, y_task, y_three, subjects, trial_id, t0_sec, stats。"""
    mat_path = Path(mat_path)
    runs = load_openbmi_mat(mat_path)
    xs_f, xs_m, yts, y3s, tids, sids, t0s = [], [], [], [], [], [], []
    tid_offset = 0
    stats_all = {
        "n_runs": len(runs),
        "n_windows": 0,
        "n_trials_kept": 0,
        "n_trials_dropped": 0,
        "subject": runs[0].subject if runs else "",
        "protocol": PROTOCOL,
        "zscore": bool(zscore),
        "blocks": ["EEG_MI_train"],
        "no_rest": True,
    }
    for eeg in runs:
        Xf, Xm, yt, y3, tid, t0, st = preprocess_run_pf1000(eeg, zscore=zscore)
        if len(yt) == 0:
            stats_all["n_trials_dropped"] += int(st.get("n_trials_dropped", 0))
            continue
        tid = tid + tid_offset
        tid_offset = int(tid.max()) + 1
        xs_f.append(Xf)
        xs_m.append(Xm)
        yts.append(yt)
        y3s.append(y3)
        tids.append(tid)
        t0s.append(t0)
        sids.append(np.array([eeg.subject] * len(yt), dtype=object))
        stats_all["n_windows"] += int(st["n_windows"])
        stats_all["n_trials_kept"] += int(st["n_trials_kept"])
        stats_all["n_trials_dropped"] += int(st["n_trials_dropped"])

    if not xs_f:
        empty = np.zeros((0, 1, 8, N_TIMES), np.float32)
        z = np.zeros((0,), np.int64)
        return (
            empty,
            empty.copy(),
            z,
            z.copy(),
            np.array([], dtype=object),
            z.copy(),
            np.zeros((0,), np.float32),
            stats_all,
        )

    X_full = np.concatenate(xs_f, axis=0)
    X_mask = np.concatenate(xs_m, axis=0)
    y_task = np.concatenate(yts, axis=0)
    y_three = np.concatenate(y3s, axis=0)
    subjects = np.concatenate(sids, axis=0)
    trial_id = np.concatenate(tids, axis=0)
    t0_sec = np.concatenate(t0s, axis=0)
    stats_all["y_three"] = np.bincount(y_three, minlength=3).tolist()
    return X_full, X_mask, y_task, y_three, subjects, trial_id, t0_sec, stats_all


def sanity_check_pf1000(X_full, X_mask, y_three) -> None:
    assert len(X_full) > 0, "没有有效窗"
    assert X_full.shape[1:] == (1, 8, N_TIMES), X_full.shape
    assert X_mask.shape == X_full.shape
    assert np.allclose(X_mask[..., -FUT:], 0.0)
    assert not np.allclose(X_full[..., -FUT:], 0.0)  # 真 future 不应全零
    assert set(np.unique(y_three)).issubset({1, 2})  # v1 无 Rest
    assert np.isfinite(X_full).all()
