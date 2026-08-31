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


def test_v3test_judge_wait_offsets_are_absolute():
    """回归：_wait_after(t0, dur)=等到 t0+dur，循环须传绝对 t_rel，禁止相邻差。"""
    cue_s, mi_s, iv = 1.0, 2.0, 0.5
    rest_s = 1.0
    rest_times = [
        round(iv * i, 3)
        for i in range(1, int(round(rest_s / iv)) + 1)
        if iv * i <= rest_s + 1e-9
    ]
    mi_times = [
        round(cue_s + iv * i, 3)
        for i in range(1, int(round((cue_s + mi_s) / iv)) + 1)
        if cue_s + iv * i <= cue_s + mi_s + 1e-9
    ]
    assert rest_times == [0.5, 1.0]
    assert mi_times == [1.5, 2.0, 2.5, 3.0]

    rest_t, cue_t = 100.0, 101.0
    # 正确：绝对偏移 → 单调墙钟目标
    correct = [rest_t + t for t in rest_times] + [cue_t + t for t in mi_times]
    assert correct == [100.5, 101.0, 102.5, 103.0, 103.5, 104.0]
    assert all(correct[i] < correct[i + 1] for i in range(len(correct) - 1))

    # 错误（旧 bug）：传相邻差 → 第二起目标塌成 anchor+0.5，判定挤在同一时刻附近
    bad = []
    prev = 0.0
    for t in mi_times:
        bad.append(cue_t + (t - prev))
        prev = t
    assert bad[0] == cue_t + 1.5
    assert bad[1:] == [cue_t + 0.5] * (len(mi_times) - 1)  # 早已过去 → 瞬间爆发


def test_v3test_run_mi_judge_spacing_not_collapsed(monkeypatch, tmp_path):
    """短跑一场 MI：各 judge 的墙钟间隔应接近 iv，不能挤在同一毫秒。"""
    import time as _time

    from experiment_game.experiment import v3test_game as vg
    from experiment_game.experiment.events_log import EventLogger
    from experiment_game.experiment.markers import MarkerPublisher
    from experiment_game.experiment.v2_config import V2Config

    clock = {"t": 0.0}
    monkeypatch.setattr(vg, "local_clock", lambda: clock["t"])

    def fake_sleep(dt):
        clock["t"] += float(dt)

    monkeypatch.setattr(_time, "sleep", fake_sleep)

    judge_wall: list[float] = []

    def judgment_fn(anchor_t, t_rel, ctx):
        judge_wall.append(clock["t"])
        return {"pred": 1, "p_max": 0.99, "gated": True}

    cfg = V2Config.load_yaml()
    cfg.v3test_rest_s = 1.0
    cfg.v3test_cue_s = 1.0
    cfg.v3test_mi_s = 2.0
    cfg.v3test_judge_interval_s = 0.5
    cfg.v3test_consecutive = 99
    cfg.v3test_n_rest = 0
    cfg.v3test_n_left = 1
    cfg.v3test_n_right = 0

    events = EventLogger(tmp_path / "events.jsonl")
    markers = MarkerPublisher(enabled=False)
    try:
        vg.run_v3test_game(
            events,
            markers,
            bridge=None,
            on_console=lambda _m: None,
            cfg=cfg,
            judgment_fn=judgment_fn,
            on_stage=lambda *_a, **_k: None,
            seed=0,
        )
    finally:
        events.close()

    # Rest 热身 2 判 + MI 4 判（1.5…3.0）
    assert len(judge_wall) >= 4
    mi_walls = judge_wall[-4:]
    gaps = [round(mi_walls[i + 1] - mi_walls[i], 2) for i in range(3)]
    assert gaps == [0.5, 0.5, 0.5], gaps


def test_create_session_dir_module_prefix(tmp_path):
    from experiment_game.experiment.session import create_session_dir

    paths = create_session_dir(tmp_path, "abctest", "w01", module_prefix="v3")
    assert paths.root.name.startswith("v3_abctest_w01_")
