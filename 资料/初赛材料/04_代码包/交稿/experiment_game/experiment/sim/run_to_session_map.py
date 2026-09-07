"""run（48 trial）→ v3 session 脚本（Rest + L/R 三分类）。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiment_game.experiment.sim.bci2a_mat_loader import (
    Bci2aRunData,
    count_run_capacity,
    load_bci2a_run,
)

LABEL_REST, LABEL_LEFT, LABEL_RIGHT = 0, 1, 2


@dataclass
class SimTrial:
    cue_sample: int
    label: int  # 0=Rest 1=L 2=R
    mat_trial_index: int  # Rest 试次为 -rest_slot_idx-1
    rest_start_sample: int
    rest_end_sample: int


@dataclass
class SimTrialScript:
    subject_id: str
    run_id: str
    mat_path: str
    fs: float
    x8: np.ndarray
    trials: List[SimTrial]
    trials_unused: List[int]
    labels_by_block: List[List[int]]
    blocks: int
    trials_per_block: int
    session_trials_total: int
    align_mode: str = "schedule_align"
    meta: Dict[str, Any] = field(default_factory=dict)


def split_block_config(session_trials_total: int, blocks: int = 2) -> Tuple[int, int]:
    """返回 (trials_per_block, blocks) 使 blocks * tpb == N 且每块 6–36。"""
    n = int(session_trials_total)
    preferred = int(blocks)
    if n < 6 or n > 48:
        raise ValueError("session_trials_total 须在 6–48")
    if preferred < 1 or preferred > 4:
        raise ValueError("blocks 须在 1–4")

    order = []
    for c in (preferred, 2, 1, 3, 4):
        if c not in order:
            order.append(c)

    for b in order:
        if n % b != 0:
            continue
        tpb = n // b
        if 6 <= tpb <= 36:
            return tpb, b
    raise ValueError(f"无法将 {n} trial 均分为合法块（每块 6–36）")


def allocate_three_class_counts(
    session_trials_total: int,
    n_l: int,
    n_r: int,
    n_rest: int,
) -> Tuple[int, int, int]:
    """均衡分配 Rest/L/R 试次数（尽量三类接近，不超过可用量）。"""
    n = int(session_trials_total)
    caps = {LABEL_REST: int(n_rest), LABEL_LEFT: int(n_l), LABEL_RIGHT: int(n_r)}
    if n > sum(caps.values()):
        raise ValueError(
            f"需要 {n} 试次，run 仅 Rest/L/R 合计 {sum(caps.values())}"
        )
    counts = {LABEL_REST: 0, LABEL_LEFT: 0, LABEL_RIGHT: 0}
    for _ in range(n):
        candidates = [c for c in (LABEL_REST, LABEL_LEFT, LABEL_RIGHT) if counts[c] < caps[c]]
        if not candidates:
            raise ValueError("Rest/L/R 可用量不足")
        lab = min(candidates, key=lambda c: (counts[c], c))
        counts[lab] += 1
    return counts[LABEL_REST], counts[LABEL_LEFT], counts[LABEL_RIGHT]


def _permute_three_class_block(rng: random.Random) -> List[int]:
    """块内 3 Rest + 3 L + 3 R 排列，避免三连同类。"""
    block = [LABEL_REST] * 3 + [LABEL_LEFT] * 3 + [LABEL_RIGHT] * 3
    for _ in range(80):
        rng.shuffle(block)
        ok = True
        for i in range(len(block) - 2):
            if block[i] == block[i + 1] == block[i + 2]:
                ok = False
                break
        if ok:
            return block
    return block


def build_three_class_block_labels(
    n_trials: int,
    rng: random.Random,
    *,
    n_rest: Optional[int] = None,
    n_left: Optional[int] = None,
    n_right: Optional[int] = None,
) -> List[int]:
    """生成 n_trials 个 Rest/L/R 标签（可选固定每类数量）。"""
    if n_trials <= 0:
        return []
    if n_rest is not None and n_left is not None and n_right is not None:
        labels = [LABEL_REST] * n_rest + [LABEL_LEFT] * n_left + [LABEL_RIGHT] * n_right
        rng.shuffle(labels)
        return labels
    if n_trials % 9 == 0:
        out: List[int] = []
        for _ in range(n_trials // 9):
            out.extend(_permute_three_class_block(rng))
        return out
    n_each = n_trials // 3
    rem = n_trials - 3 * n_each
    labels = [LABEL_REST] * n_each + [LABEL_LEFT] * n_each + [LABEL_RIGHT] * n_each
    for lab in (LABEL_LEFT, LABEL_RIGHT, LABEL_REST):
        if rem <= 0:
            break
        labels.append(lab)
        rem -= 1
    rng.shuffle(labels)
    return labels


def build_sim_script(
    mat_path: Path | str,
    run_id: str,
    *,
    session_trials_total: int = 36,
    blocks: int = 2,
    align_mode: str = "schedule_align",
    seed: Optional[int] = None,
    rest_s: float = 4.0,
) -> SimTrialScript:
    """构建仿真 session 脚本（Rest/L/R 三分类，静息计入试次）。"""
    rd = load_bci2a_run(mat_path, run_id, rest_s=rest_s)
    n_l, n_r, n_rest, n_max = count_run_capacity(rd)

    n_rest_t, n_l_t, n_r_t = allocate_three_class_counts(
        session_trials_total, n_l, n_r, n_rest
    )
    n = n_rest_t + n_l_t + n_r_t

    tpb, b = split_block_config(n, blocks)
    rng = random.Random(seed)

    # 按时间序取 MI 试次池
    mi_indices = list(range(len(rd.labels)))
    left_pool = [i for i in mi_indices if int(rd.labels[i]) == LABEL_LEFT][:n_l_t]
    right_pool = [i for i in mi_indices if int(rd.labels[i]) == LABEL_RIGHT][:n_r_t]
    rest_pool = list(range(min(n_rest, n_rest_t)))

    if len(left_pool) < n_l_t or len(right_pool) < n_r_t or len(rest_pool) < n_rest_t:
        raise ValueError(
            f"run {run_id} 可用 Rest/L/R 不足 "
            f"(需要 R{n_rest_t}/L{n_l_t}/R{n_r_t}，"
            f"有 Rest{len(rest_pool)}/L{len(left_pool)}/R{len(right_pool)})"
        )

    mi_used = set(left_pool) | set(right_pool)
    trials: List[SimTrial] = []
    labels_by_block: List[List[int]] = []
    rest_len = int(round(rest_s * rd.fs))

    all_labels = build_three_class_block_labels(
        n, rng, n_rest=n_rest_t, n_left=n_l_t, n_right=n_r_t
    )

    label_idx = 0
    for bi in range(b):
        block_labels = all_labels[label_idx : label_idx + tpb]
        label_idx += tpb
        block_label_list: List[int] = []
        for lab in block_labels:
            if lab == LABEL_REST:
                ri = rest_pool.pop(0)
                rs = int(rd.rest_start_samples[ri])
                cue = int(rd.rest_end_samples[ri])
                trials.append(
                    SimTrial(
                        cue_sample=cue,
                        label=LABEL_REST,
                        mat_trial_index=-(ri + 1),
                        rest_start_sample=rs,
                        rest_end_sample=cue,
                    )
                )
            elif lab == LABEL_LEFT:
                j = left_pool.pop(0)
                cue = int(rd.cue_samples[j])
                trials.append(
                    SimTrial(
                        cue_sample=cue,
                        label=LABEL_LEFT,
                        mat_trial_index=int(rd.mat_trial_indices[j]),
                        rest_start_sample=max(0, cue - rest_len),
                        rest_end_sample=cue,
                    )
                )
            else:
                j = right_pool.pop(0)
                cue = int(rd.cue_samples[j])
                trials.append(
                    SimTrial(
                        cue_sample=cue,
                        label=LABEL_RIGHT,
                        mat_trial_index=int(rd.mat_trial_indices[j]),
                        rest_start_sample=max(0, cue - rest_len),
                        rest_end_sample=cue,
                    )
                )
            block_label_list.append(int(lab))
        labels_by_block.append(block_label_list)

    unused_mi = [int(rd.mat_trial_indices[i]) for i in mi_indices if i not in mi_used]

    return SimTrialScript(
        subject_id=rd.subject_id,
        run_id=rd.run_id,
        mat_path=rd.mat_path,
        fs=rd.fs,
        x8=rd.x8,
        trials=trials,
        trials_unused=unused_mi,
        labels_by_block=labels_by_block,
        blocks=b,
        trials_per_block=tpb,
        session_trials_total=n,
        align_mode=str(align_mode),
        meta={
            "run_index": rd.run_index,
            "n_lr_available": n_l + n_r,
            "n_rest_available": n_rest,
            "n_total_available": n_max,
            "n_rest_used": n_rest_t,
            "n_left_used": n_l_t,
            "n_right_used": n_r_t,
            "include_rest": True,
            "three_class": True,
        },
    )


def build_sim_script_from_labels(
    mat_path: Path | str,
    run_id: str,
    labels: List[int],
    *,
    blocks: int = 2,
    align_mode: str = "schedule_align",
    rest_s: float = 4.0,
) -> SimTrialScript:
    """按已知标签序列从 mat 池顺序 pop（旧 session 无 seed 时重建）。"""
    rd = load_bci2a_run(mat_path, run_id, rest_s=rest_s)
    n_l, n_r, n_rest, n_max = count_run_capacity(rd)
    labs = [int(x) for x in labels]
    n = len(labs)
    n_rest_t = labs.count(LABEL_REST)
    n_l_t = labs.count(LABEL_LEFT)
    n_r_t = labs.count(LABEL_RIGHT)
    tpb, b = split_block_config(n, blocks)
    mi_indices = list(range(len(rd.labels)))
    left_pool = [i for i in mi_indices if int(rd.labels[i]) == LABEL_LEFT][:n_l_t]
    right_pool = [i for i in mi_indices if int(rd.labels[i]) == LABEL_RIGHT][:n_r_t]
    rest_pool = list(range(min(n_rest, n_rest_t)))
    if len(left_pool) < n_l_t or len(right_pool) < n_r_t or len(rest_pool) < n_rest_t:
        raise ValueError(
            f"run {run_id} 标签序列无法匹配 mat 池 "
            f"(需要 R{n_rest_t}/L{n_l_t}/R{n_r_t})"
        )
    mi_used = set(left_pool) | set(right_pool)
    trials: List[SimTrial] = []
    labels_by_block: List[List[int]] = []
    rest_len = int(round(rest_s * rd.fs))
    for bi in range(b):
        block_labels = labs[bi * tpb : (bi + 1) * tpb]
        block_label_list: List[int] = []
        for lab in block_labels:
            if lab == LABEL_REST:
                ri = rest_pool.pop(0)
                rs = int(rd.rest_start_samples[ri])
                cue = int(rd.rest_end_samples[ri])
                trials.append(
                    SimTrial(
                        cue_sample=cue,
                        label=LABEL_REST,
                        mat_trial_index=-(ri + 1),
                        rest_start_sample=rs,
                        rest_end_sample=cue,
                    )
                )
            elif lab == LABEL_LEFT:
                j = left_pool.pop(0)
                cue = int(rd.cue_samples[j])
                trials.append(
                    SimTrial(
                        cue_sample=cue,
                        label=LABEL_LEFT,
                        mat_trial_index=int(rd.mat_trial_indices[j]),
                        rest_start_sample=max(0, cue - rest_len),
                        rest_end_sample=cue,
                    )
                )
            else:
                j = right_pool.pop(0)
                cue = int(rd.cue_samples[j])
                trials.append(
                    SimTrial(
                        cue_sample=cue,
                        label=LABEL_RIGHT,
                        mat_trial_index=int(rd.mat_trial_indices[j]),
                        rest_start_sample=max(0, cue - rest_len),
                        rest_end_sample=cue,
                    )
                )
            block_label_list.append(int(lab))
        labels_by_block.append(block_label_list)
    unused_mi = [int(rd.mat_trial_indices[i]) for i in mi_indices if i not in mi_used]
    return SimTrialScript(
        subject_id=rd.subject_id,
        run_id=rd.run_id,
        mat_path=rd.mat_path,
        fs=rd.fs,
        x8=rd.x8,
        trials=trials,
        trials_unused=unused_mi,
        labels_by_block=labels_by_block,
        blocks=b,
        trials_per_block=tpb,
        session_trials_total=n,
        align_mode=str(align_mode),
        meta={
            "run_index": rd.run_index,
            "n_lr_available": n_l + n_r,
            "n_rest_available": n_rest,
            "n_total_available": n_max,
            "n_rest_used": n_rest_t,
            "n_left_used": n_l_t,
            "n_right_used": n_r_t,
            "include_rest": True,
            "three_class": True,
            "from_labels": True,
        },
    )
