"""OpenBMI · past+cur+future 1000pt 预处理臂（新路径，不改旧 openbmi pipeline）。

协议：资料/.../数据切片与边界过滤说明.md

三类状态切割（冻结，@250 Hz）：
────────────────────────────────────────────────────────────────────────
标签：rest/idle → y_three=0,y_task=0；left→1；right→2（y_task=1）

共用几何（相对「段原点」= Task 的 cue / Rest 的 rest_origin）：
  past=100 (0.4s) | cur=500 (2.0s) | future=400 (1.6s) → 共 1000 点
  评分区 SCORE=4.0s：仅 cur 完全落入 [0, SCORE) 的锚点保留
  合法 t0 ∈ {0.4, 0.5, …, 2.0}（hop=0.1）；缺 past / 缺 future / 越评分区 → 裁窗
  段长 SEG = SCORE + POST = 5.6s；基线 = 段原点前 0.5s（校正后丢弃）

【Left / Right · Task】
  连续段（相对 cue）：[cue, cue+5.6s)  —— 不读 cue 前任何样本
  基线：段首 0.5s 均值减全段（不丢点）；seg = MI 评分 4s + post-MI 1.6s
  future 取自 post-MI（不得零填）

【Rest / 空闲】
  源区间：每个非首个 cue 的「Cue 前」满 5.6s（与上一 trial MI [prev_cue, prev_cue+4s)
  重叠则起点右移；不足 5.6s → 丢弃该 Rest trial）
  rest_origin = cue − 5.6s；连续段 [rest_origin−0.5s, cue)
  丢基线后 seg：[rest_origin, cue) = Rest 评分 4s + Rest-post 1.6s
  ※ 整段落在 Cue 之前，future 不进入后续 MI
  数量上限：max_rest = min(n_left_cues, n_right_cues)（与旧 hop100 平衡一致）

输出：(N,1,8,1000) X_full / X_mask；y_three ∈ {0,1,2}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample

from src.common.eeg_types import ContinuousEEG
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.resample_zscore import to_model_tensor, trial_zscore
from src.common.steps.select_channels import select_channels
from src.datasets.bci2a.labels import filter_left_right_events
from src.datasets.openbmi.load_mat import load_openbmi_mat

PROTOCOL = "openbmi_2s_hop100_pf1000"
# 含 Rest + Task 自 cue 起切；旧 shard 失效
PROTOCOL_VERSION = 3

FS_OUT = 250.0
PAST = 100
CUR = 500
FUT = 400
N_TIMES = PAST + CUR + FUT  # 1000
HOP = 25  # 0.1 s @ 250 Hz
MI_SEC = 4.0  # Task / Rest 共用评分区长度
POST_SEC = 1.6
SEG_SEC = MI_SEC + POST_SEC  # 5.6
BASELINE_SEC = 0.5
MI_END_PTS = int(round(MI_SEC * FS_OUT))  # 1000

# 冻结切割定义（写入 meta / 文档对齐）
CUT_SPEC: dict[str, Any] = {
    "fs_out": FS_OUT,
    "geometry": {
        "past_pts": PAST,
        "cur_pts": CUR,
        "future_pts": FUT,
        "total_pts": N_TIMES,
        "hop_pts": HOP,
        "score_sec": MI_SEC,
        "post_sec": POST_SEC,
        "seg_sec": SEG_SEC,
        "baseline_sec": BASELINE_SEC,
        "legal_t0_sec": [round(0.4 + 0.1 * i, 1) for i in range(17)],
    },
    "label_map": {"rest": 0, "left": 1, "right": 2},
    "task": {
        "name": "left_right",
        "seg_rel_cue_sec": [0.0, SEG_SEC],
        "score_rel_cue_sec": [0.0, MI_SEC],
        "post_rel_cue_sec": [MI_SEC, SEG_SEC],
        "baseline": "in_segment_first_0.5s_mean_subtract_keep_all",
        "note": "从 cue 起切 [0, 5.6)；不读 cue 前；段首 0.5s 仅作基线均值",
    },
    "rest": {
        "name": "idle",
        "source": "cue_before_full_5.6s",
        "seg_rel_cue_sec": [-SEG_SEC - BASELINE_SEC, 0.0],
        "score_rel_origin_sec": [0.0, MI_SEC],
        "post_rel_origin_sec": [MI_SEC, SEG_SEC],
        "rest_origin": "cue - 5.6s",
        "balance": "max_rest = min(n_left_cues, n_right_cues)",
        "note": "整段在 Cue 前；评分+post 同 Task 几何；future 不进入 MI",
    },
}


def _extract_seg_task_from_cue(
    x: np.ndarray,
    cue: int,
    fs: float,
    *,
    seg_sec: float = SEG_SEC,
    baseline_sec: float = BASELINE_SEC,
) -> np.ndarray | None:
    """Task：只取 [cue, cue+seg_sec)；不读 cue 前。段首 baseline_sec 均值减全段（不丢点）。"""
    n_seg = int(round(seg_sec * fs))
    n_base = int(round(baseline_sec * fs))
    t0 = int(cue)
    t1 = t0 + n_seg
    if t0 < 0 or t1 > x.shape[0]:
        return None
    seg = x[t0:t1].astype(np.float64, copy=True)
    if seg.shape[0] != n_seg:
        return None
    if n_base > 0:
        base = seg[:n_base].mean(axis=0, keepdims=True)
        seg = seg - base
    return seg


def _extract_seg_rest_from_origin(
    x: np.ndarray,
    origin: int,
    fs: float,
    *,
    seg_sec: float = SEG_SEC,
    baseline_sec: float = BASELINE_SEC,
) -> np.ndarray | None:
    """Rest：原点前 baseline + 原点后 seg_sec；基线校正后丢掉基线 → 从原点起。"""
    n_base = int(round(baseline_sec * fs))
    n_seg = int(round(seg_sec * fs))
    t0 = int(origin) - n_base
    t1 = int(origin) + n_seg
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
    seg250: (T>=1400, 8) 从段原点起。
    返回 list of (x_full_T8 (1000,8), y_task, y_three, t0_sec)。
    """
    if seg250.shape[0] < int(round(SEG_SEC * FS_OUT)):
        return []
    out: list[tuple[np.ndarray, int, int, float]] = []
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
        out.append(
            (x_full.astype(np.float32), int(lab_task), int(lab_three), float(i0) / FS_OUT)
        )
    return out


def iter_rest_origins_pf1000(
    cue_samples: np.ndarray,
    fs: float,
    n_times: int,
    *,
    score_sec: float = MI_SEC,
    post_sec: float = POST_SEC,
    task_sec: float = MI_SEC,
) -> list[int]:
    """
    Rest 段原点（样本下标）：每个非首个 cue，要求 Cue 前满 score+post 秒，
    且不与上一 trial MI [prev_cue, prev_cue+task_sec) 重叠（重叠则丢弃，不缩短）。
    返回 rest_origin 列表；连续段为 [origin, cue) 长恰好 SEG_SEC。
    """
    seg_len = int(round((score_sec + post_sec) * fs))
    task_len = int(round(task_sec * fs))
    cues = np.sort(np.asarray(cue_samples, dtype=int).reshape(-1))
    out: list[int] = []
    for i in range(1, len(cues)):
        cue = int(cues[i])
        origin = cue - seg_len
        prev_task_end = int(cues[i - 1]) + task_len
        if origin < 0 or cue > n_times:
            continue
        # 与上一 MI 重叠 → 不缩短（缩短会破坏 5.6s 齐全几何）
        if origin < prev_task_end:
            continue
        out.append(origin)
    return out


def preprocess_run_pf1000(
    eeg: ContinuousEEG,
    *,
    zscore: bool = True,
    add_rest: bool = True,
    max_rest: int | None = None,
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
    n_rest_src = 0
    n_rest_kept = 0
    n_rest_drop = 0

    # ── Left / Right ──
    for cue, lab_task, lab_three, _ in kept:
        seg = _extract_seg_task_from_cue(x, int(cue), eeg.fs)
        if seg is None:
            n_drop_trial += 1
            continue
        seg250 = _resample_seg(seg, eeg.fs)
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

    # ── Rest / 空闲 ──
    if add_rest and len(kept) > 0:
        origins = iter_rest_origins_pf1000(kept[:, 0], eeg.fs, x.shape[0])
        n_rest_src = len(origins)
        if max_rest is None:
            n_left = int(np.sum(kept[:, 2] == 1))
            n_right = int(np.sum(kept[:, 2] == 2))
            max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        origins = origins[: int(max_rest)]
        for origin in origins:
            seg = _extract_seg_rest_from_origin(x, int(origin), eeg.fs)
            if seg is None:
                n_rest_drop += 1
                continue
            seg250 = _resample_seg(seg, eeg.fs)
            n_cand_anchor += len(range(0, MI_END_PTS - CUR + 1, HOP))
            wins = windows_from_seg250(
                seg250, lab_task=0, lab_three=0, zscore=zscore
            )
            if not wins:
                n_rest_drop += 1
                continue
            for x_full, yt, y3, t0 in wins:
                xs_full.append(x_full)
                y_task.append(yt)
                y_three.append(y3)
                trial_ids.append(tid)
                t0s.append(t0)
                n_keep_win += 1
            tid += 1
            n_rest_kept += 1

    stats = {
        "n_cues": int(len(kept)),
        "n_trials_kept": int(tid),
        "n_trials_dropped": int(n_drop_trial),
        "n_cand_anchors": int(n_cand_anchor),
        "n_windows": int(n_keep_win),
        "n_rest_sources": int(n_rest_src),
        "n_rest_trials_kept": int(n_rest_kept),
        "n_rest_trials_dropped": int(n_rest_drop),
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "zscore": bool(zscore),
        "no_rest": not bool(add_rest),
        "add_rest": bool(add_rest),
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
    add_rest: bool = True,
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
        "n_rest_trials_kept": 0,
        "subject": runs[0].subject if runs else "",
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "zscore": bool(zscore),
        "blocks": ["EEG_MI_train"],
        "no_rest": not bool(add_rest),
        "add_rest": bool(add_rest),
    }
    for eeg in runs:
        Xf, Xm, yt, y3, tid, t0, st = preprocess_run_pf1000(
            eeg, zscore=zscore, add_rest=add_rest
        )
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
        stats_all["n_rest_trials_kept"] += int(st.get("n_rest_trials_kept", 0))

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
    stats_all["y_task"] = np.bincount(y_task, minlength=2).tolist()
    return X_full, X_mask, y_task, y_three, subjects, trial_id, t0_sec, stats_all


def sanity_check_pf1000(X_full, X_mask, y_three, y_task=None) -> None:
    assert len(X_full) > 0, "没有有效窗"
    assert X_full.shape[1:] == (1, 8, N_TIMES), X_full.shape
    assert X_mask.shape == X_full.shape
    assert np.allclose(X_mask[..., -FUT:], 0.0)
    assert not np.allclose(X_full[..., -FUT:], 0.0)  # 真 future 不应全零
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    if y_task is not None:
        y_task = np.asarray(y_task)
        assert set(np.unique(y_task)).issubset({0, 1})
        assert np.all((y_three == 0) == (y_task == 0))
        assert np.all(y_task[y_three > 0] == 1)
    assert np.isfinite(X_full).all()
