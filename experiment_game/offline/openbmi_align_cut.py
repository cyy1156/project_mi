"""OpenBMI-Align 离线切窗（与 ft_subject_from_v3 openbmi_align / preprocess_run_3s_hop100 同构）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from src.common.steps.epoch_baseline import task_window_cue_0_to_4  # noqa: E402
from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402
from src.common.steps.slide_1s import extract_segment_baseline, iter_rest_sources_cue_before  # noqa: E402
from src.common.steps.slide_3s_hop100 import WIN_SEC as WIN_SEC_3S, segment_to_3s_hop100_windows  # noqa: E402

FS = 250.0
# 2026-08-29 冻结：与 experiment/channel_layout.DEVICE_CHANNEL_LABELS 统一（设备序=模型序）
FROZEN = ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"]


def cue_time_from_row(r: Dict[str, Any]) -> Optional[float]:
    tc, tm = r.get("t_cue"), r.get("t_mi_start")
    if tc not in (None, "") and tm not in (None, ""):
        tc_f, tm_f = float(tc), float(tm)
        if abs(tm_f - tc_f) <= 0.5:
            return tc_f
        return tm_f
    if tc not in (None, ""):
        return float(tc)
    if tm not in (None, ""):
        return float(tm)
    return None


def _lsl_to_sample(t_lsl: np.ndarray, t: float) -> int:
    return int(np.searchsorted(t_lsl, float(t)))


def iter_rest_sources_from_table(
    rows: List[Dict[str, Any]],
    t_lsl: np.ndarray,
    *,
    skip_rejected: bool = True,
    skip_invalid: bool = True,
    min_win_sec: float = WIN_SEC_3S,
) -> List[Tuple[int, int, int]]:
    """从 trial_table 的 t_rest_start/t_rest_end 取 Rest 段（同 trial_id，Cue 前纯静息）。"""
    min_len = int(round(min_win_sec * FS))
    out: List[Tuple[int, int, int]] = []
    for r in rows:
        if skip_rejected and str(r.get("rejected") or "0") == "1":
            continue
        if skip_invalid and str(r.get("invalid") or "0") == "1":
            continue
        ts, te = r.get("t_rest_start"), r.get("t_rest_end")
        if ts in (None, "") or te in (None, ""):
            continue
        t0 = _lsl_to_sample(t_lsl, float(ts))
        t1 = _lsl_to_sample(t_lsl, float(te))
        if t1 - t0 < min_len:
            continue
        try:
            tid_raw = r.get("trial_id")
            if tid_raw is None or (isinstance(tid_raw, float) and tid_raw != tid_raw):
                tid = 0
            else:
                tid = int(float(tid_raw))
        except (TypeError, ValueError):
            tid = 0
        out.append((tid, t0, t1))
    out.sort(key=lambda x: (x[1], x[0]))
    return out


def _wins_to_model(wins: List[np.ndarray]) -> List[np.ndarray]:
    return [w.T.astype(np.float32) for w in wins if w.shape[0] == int(round(WIN_SEC_3S * FS))]


def cut_openbmi_align_from_table(
    x_filt: np.ndarray,
    t_lsl: np.ndarray,
    rows: List[Dict[str, Any]],
    *,
    task_phases: Optional[set] = None,
    include_rest_interval: bool = True,
    skip_invalid: bool = True,
    skip_rejected: bool = True,
) -> Tuple[List[np.ndarray], List[int], List[int], List[int]]:
    """从 trial_table 行切 OpenBMI 对齐窗。返回 wins, y_three, y_task, trial_ids。"""
    wins: List[np.ndarray] = []
    y_three: List[int] = []
    y_task: List[int] = []
    tids: List[int] = []

    task_rows: List[Dict[str, Any]] = []
    cue_samples: List[int] = []
    n_left = n_right = 0

    for r in rows:
        if skip_rejected and str(r.get("rejected") or "0") == "1":
            continue
        if skip_invalid and str(r.get("invalid") or "0") == "1":
            continue
        phase = str(r.get("phase") or "")
        if task_phases is not None and phase and phase not in task_phases:
            continue
        lab = int(r.get("label") or -1)
        if lab not in (1, 2):
            continue
        t_cue = cue_time_from_row(r)
        if t_cue is None:
            continue
        cue_idx = _lsl_to_sample(t_lsl, t_cue)
        seg = task_window_cue_0_to_4(x_filt, cue_idx, FS)
        if seg is None:
            continue
        seg_wins = segment_to_3s_hop100_windows(seg, FS, zscore=True)
        ws = _wins_to_model(seg_wins)
        if not ws:
            continue
        tid = int(r["trial_id"])
        task_rows.append(r)
        cue_samples.append(cue_idx)
        if lab == 1:
            n_left += 1
        else:
            n_right += 1
        for w in ws:
            wins.append(w)
            y_three.append(lab)
            y_task.append(1)
            tids.append(tid)

    if include_rest_interval:
        max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        rest_sources = iter_rest_sources_from_table(
            rows,
            t_lsl,
            skip_rejected=skip_rejected,
            skip_invalid=skip_invalid,
            min_win_sec=WIN_SEC_3S,
        )
        if not rest_sources and cue_samples:
            # 无 rest 打点时回退（OpenBMI .mat / 历史 session）
            fb = iter_rest_sources_cue_before(
                np.asarray(sorted(cue_samples), dtype=int),
                FS,
                x_filt.shape[0],
                rest_sec=4.0,
                task_sec=4.0,
                min_win_sec=WIN_SEC_3S,
            )
            rest_sources = [(-(i + 1), int(t0), int(t1)) for i, (t0, t1) in enumerate(fb)]
        for ri, (tid, t0, t1) in enumerate(rest_sources[: int(max_rest)]):
            seg = extract_segment_baseline(x_filt, int(t0), int(t1), FS, baseline_sec=0.5)
            if seg is None:
                continue
            seg_wins = segment_to_3s_hop100_windows(seg, FS, zscore=True)
            ws = _wins_to_model(seg_wins)
            for w in ws:
                wins.append(w)
                y_three.append(0)
                y_task.append(0)
                tids.append(int(tid) if tid > 0 else -(ri + 1))

    return wins, y_three, y_task, tids
