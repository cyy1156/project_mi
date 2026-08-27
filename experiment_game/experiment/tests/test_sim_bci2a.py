"""BCI2a 仿真映射与回放单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_MAT = _REPO / "DATA" / "bci2a" / "A01T.mat"


@pytest.mark.skipif(not _MAT.is_file(), reason="需要 DATA/bci2a/A01T.mat")
def test_build_sim_script_36_with_rest():
    from experiment_game.experiment.sim.run_to_session_map import build_sim_script

    script = build_sim_script(_MAT, "run3", session_trials_total=36, blocks=2, seed=42)
    assert script.session_trials_total == 36
    assert script.blocks == 2
    assert script.trials_per_block == 18
    labels = [t.label for t in script.trials]
    assert labels.count(0) + labels.count(1) + labels.count(2) == 36
    assert labels.count(1) == 12
    assert labels.count(2) == 11
    assert script.meta["three_class"] is True


@pytest.mark.skipif(not _MAT.is_file(), reason="需要 DATA/bci2a/A01T.mat")
def test_build_sim_script_18_trials():
    from experiment_game.experiment.sim.run_to_session_map import build_sim_script

    script = build_sim_script(_MAT, "run3", session_trials_total=18, blocks=2, seed=42)
    assert script.session_trials_total == 18
    assert script.blocks == 2
    assert script.trials_per_block == 9
    labels = [t.label for t in script.trials]
    assert len(labels) == 18
    assert labels.count(0) == labels.count(1) == labels.count(2) == 6


@pytest.mark.skipif(not _MAT.is_file(), reason="需要 DATA/bci2a/A01T.mat")
def test_replay_timeline_length():
    from experiment_game.experiment.sim.bci2a_replay_source import build_schedule_align_timeline
    from experiment_game.experiment.sim.run_to_session_map import build_sim_script

    script = build_sim_script(_MAT, "run3", session_trials_total=6, blocks=1, seed=1)
    tl = build_schedule_align_timeline(script, rest_s=4, prep_s=2, mi_s=4, iti_s=3)
    # Rest 试次跳过首段 4s rest（与范式一致）：L/R=13s，Rest=9s
    n_rest = sum(1 for t in script.trials if int(t.label) == 0)
    n_lr = len(script.trials) - n_rest
    expect = (n_lr * 13 + n_rest * 9) * 250
    assert tl.shape == (expect, 8)
    assert np.isfinite(tl).all()


@pytest.mark.skipif(not _MAT.is_file(), reason="需要 DATA/bci2a/A01T.mat")
def test_timing_align_timeline():
    from experiment_game.experiment.sim.bci2a_replay_source import (
        build_replay_timeline,
        build_schedule_align_timeline,
    )
    from experiment_game.experiment.sim.run_to_session_map import build_sim_script

    script = build_sim_script(_MAT, "run3", session_trials_total=6, blocks=1, seed=1)
    tl_sched = build_schedule_align_timeline(script, rest_s=4, prep_s=2, mi_s=4, iti_s=3)
    tl_time = build_replay_timeline(
        script, align_mode="timing_align", rest_s=4, prep_s=2, mi_s=4, iti_s=3
    )
    assert tl_sched.shape[1] == 8 and tl_time.shape[1] == 8
    assert len(tl_time) >= len(tl_sched)
    assert np.isfinite(tl_time).all()


def test_split_block_config():
    from experiment_game.experiment.sim.run_to_session_map import split_block_config

    tpb, b = split_block_config(36, 2)
    assert tpb == 18 and b == 2
    tpb, b = split_block_config(48, 2)
    assert tpb == 24 and b == 2


def test_allocate_three_class():
    from experiment_game.experiment.sim.run_to_session_map import allocate_three_class_counts

    nr, nl, nrgt = allocate_three_class_counts(36, 12, 11, 23)
    assert nr + nl + nrgt == 36
    assert nl == 12 and nrgt == 11
    nr, nl, nrgt = allocate_three_class_counts(18, 12, 11, 23)
    assert nr == nl == nrgt == 6


@pytest.mark.skipif(not _MAT.is_file(), reason="需要 DATA/bci2a/A01T.mat")
def test_sim_script_no_duplicate_pool_pops():
    """L/R MI 位置与 Rest 槽位在各自池内不得重复 pop。"""
    from experiment_game.experiment.sim.bci2a_mat_loader import load_bci2a_run
    from experiment_game.experiment.sim.run_to_session_map import build_sim_script

    script = build_sim_script(_MAT, "run3", session_trials_total=36, blocks=2, seed=42)
    rd = load_bci2a_run(_MAT, "run3")

    flat_labels = [x for b in script.labels_by_block for x in b]
    trial_labels = [t.label for t in script.trials]
    assert flat_labels == trial_labels

    lr_mi_pos: list[int] = []
    rest_slots: list[int] = []
    for t in script.trials:
        if t.label == 0:
            rest_slots.append(-t.mat_trial_index - 1)
        elif t.label in (1, 2):
            for i, mid in enumerate(rd.mat_trial_indices):
                if int(mid) == t.mat_trial_index:
                    lr_mi_pos.append(int(i))
                    assert int(rd.labels[i]) == t.label
                    break
            else:
                pytest.fail(f"mat_trial_index {t.mat_trial_index} 不在 run 内")

    assert len(lr_mi_pos) == len(set(lr_mi_pos)), f"L/R MI 位置重复: {lr_mi_pos}"
    assert len(rest_slots) == len(set(rest_slots)), f"Rest 槽位重复: {rest_slots}"


@pytest.mark.skipif(not _MAT.is_file(), reason="需要 DATA/bci2a/A01T.mat")
def test_sim_script_lr_mat_index_unique():
    """同一 mat trial 编号不得被两次 L/R 试次使用。"""
    from experiment_game.experiment.sim.run_to_session_map import build_sim_script

    script = build_sim_script(_MAT, "run3", session_trials_total=36, blocks=2, seed=42)
    lr_mat_ids = [t.mat_trial_index for t in script.trials if t.label in (1, 2)]
    assert len(lr_mat_ids) == len(set(lr_mat_ids))


@pytest.mark.skipif(not _MAT.is_file(), reason="需要 DATA/bci2a/A01T.mat")
def test_build_sim_script_from_labels_matches_seed():
    from experiment_game.experiment.sim.run_to_session_map import (
        build_sim_script,
        build_sim_script_from_labels,
    )

    a = build_sim_script(_MAT, "run3", session_trials_total=36, blocks=2, seed=7)
    labels = [t.label for t in a.trials]
    b = build_sim_script_from_labels(_MAT, "run3", labels, blocks=2)
    assert [(t.cue_sample, t.label, t.mat_trial_index) for t in a.trials] == [
        (t.cue_sample, t.label, t.mat_trial_index) for t in b.trials
    ]


def test_leave_next_train_runs():
    from experiment_game.experiment.sim.ramp import leave_next_train_runs, ramp_stage

    m = {
        "session_queue": ["run3", "run4", "run5"],
        "sessions_completed": [
            {"run_id": "run3", "session_dir": "/s/run3"},
            {"run_id": "run4", "session_dir": "/s/run4"},
        ],
    }
    assert ramp_stage(m, "run5") == 2
    train = leave_next_train_runs(m, "run5")
    assert [r for r, _ in train] == ["run3", "run4"]
    assert leave_next_train_runs(m, "run3") == []
