"""v2 游戏测试（v3 权重 · 20 试次 · 满分 15）单元测试。"""

from __future__ import annotations

from experiment_game.experiment.v3test_game import (
    StreakCounter,
    build_v3test_schedule,
)


def test_schedule_counts_and_no_quad_repeat():
    sch = build_v3test_schedule(10, 5, 5, rng=__import__("random").Random(0))
    assert len(sch) == 20
    assert sch.count(0) == 10
    assert sch.count(1) == 5
    assert sch.count(2) == 5
    for i in range(len(sch) - 3):
        assert not (sch[i] == sch[i + 1] == sch[i + 2] == sch[i + 3])


def test_streak_hit_and_reset():
    st = StreakCounter(1, need=5)
    assert st.feed(1) == 1
    assert st.feed(1) == 2
    assert st.feed(2) == 0  # 错类清零
    for _ in range(4):
        st.feed(1)
    assert not st.hit
    st.feed(1)
    assert st.hit
    assert st.max_count >= 5


def test_streak_signal_bad_resets_via_none():
    st = StreakCounter(0, need=5)
    for _ in range(3):
        st.feed(0)
    st.feed(None)
    assert st.count == 0


def test_v2_config_v3_test_score_max():
    from experiment_game.experiment.v2_config import V2Config

    cfg = V2Config.load_yaml()
    assert cfg.game_mode == "v3_test"
    mx = (
        cfg.v3test_n_rest * cfg.v3test_rest_points
        + (cfg.v3test_n_left + cfg.v3test_n_right) * cfg.v3test_mi_points
    )
    assert mx == 15.0


def test_v3_yaml_cue_and_block_gap():
    from experiment_game.experiment.v3_config import V3Config

    cfg = V3Config.load_yaml()
    assert abs(cfg.cue_s - 1.0) < 1e-9
    assert abs(cfg.block_gap_s - 30.0) < 1e-9


def test_create_session_dir_module_prefix(tmp_path):
    from experiment_game.experiment.session import create_session_dir

    paths = create_session_dir(tmp_path, "abctest", "w01", module_prefix="v3")
    assert paths.root.name.startswith("v3_abctest_w01_")
