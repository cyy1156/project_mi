"""范式改动自测（配置 / 前缀 / 接线 / v3test 干跑）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    from experiment_game.experiment.v2_config import V2Config
    from experiment_game.experiment.v3_config import V3Config

    v2, v3 = V2Config.load_yaml(), V3Config.load_yaml()
    assert v2.game_mode == "v3_test", v2.game_mode
    assert v2.use_v3_weights is True
    assert abs(v2.v3test_rest_s - 5) < 1e-9
    assert abs(v2.v3test_cue_s - 1) < 1e-9
    assert abs(v2.v3test_mi_s - 10) < 1e-9
    assert v2.v3test_n_rest == 10 and v2.v3test_n_left == 5 and v2.v3test_n_right == 5
    mx = (
        v2.v3test_n_rest * v2.v3test_rest_points
        + (v2.v3test_n_left + v2.v3test_n_right) * v2.v3test_mi_points
    )
    assert mx == 15.0, mx
    assert abs(v3.cue_s - 1.0) < 1e-9 and abs(v3.block_gap_s - 30) < 1e-9
    print("OK config", "v2_score_max", mx, "v3 cue/gap", v3.cue_s, v3.block_gap_s)

    from experiment_game.experiment.session import create_session_dir
    from experiment_game.experiment.subject_registry import (
        _parse_session_name,
        _split_module_prefix,
    )

    with tempfile.TemporaryDirectory() as td:
        for pref in ("v1", "v2", "v3", "v4", "sim"):
            p = create_session_dir(Path(td), "abctest", "w01", module_prefix=pref)
            assert p.root.name.startswith(f"{pref}_abctest_w01_"), p.root.name
        mod, rest = _split_module_prefix("v3_abctest_w01_20260830_120000")
        assert mod == "v3" and rest.startswith("abctest_")
        sid, sess, _stamp = _parse_session_name("v3_abctest_w01_20260830_120000")
        assert sid == "abctest" and sess == "w01"
    print("OK module_prefix + parse")

    from experiment_game.experiment.registry_factory import (
        _use_subject_weights,
        verify_subject_weight_paths,
    )

    class C:
        pass

    c = C()
    c.use_v3_weights = True
    c.subject_models_dir = ""
    assert _use_subject_weights(c) is False
    c.subject_models_dir = "D:/no/such"
    assert _use_subject_weights(c) is True
    errs = verify_subject_weight_paths("D:/no/such")
    assert errs and ("缺" in errs[0] or "fail" in errs[0].lower() or "解析" in errs[0])
    print("OK registry_factory guards:", errs[0][:60])

    from experiment_game.experiment.v3test_game import (
        StreakCounter,
        build_v3test_schedule,
        run_v3test_game,
    )

    sch = build_v3test_schedule(10, 5, 5)
    assert len(sch) == 20
    st = StreakCounter(2, need=5)
    for _ in range(5):
        st.feed(2)
    assert st.hit
    print("OK v3test schedule/streak")

    html = (_REPO / "experiment_game/web/operator.html").read_text(encoding="utf-8")
    js = (_REPO / "experiment_game/web/js/operator.js").read_text(encoding="utf-8")
    bridge = (_REPO / "experiment_game/web/js/v2_bridge.js").read_text(encoding="utf-8")
    for needle in ("btn-ft-all-v3", "btn-model-eval-grid", "model-eval-result", "v2 游戏测试"):
        assert needle in html, needle
    for needle in (
        "btnFtAllV3",
        "btnModelEvalGrid",
        "sendFinetuneStart",
        "renderModelEvalResult",
        "model_eval_grid",
        "model_eval_result",
        "phase_mode !== \"v3_session\"",
    ):
        assert needle in js, needle
    assert "cueSplit" in (Path(_REPO) / "experiment_game/web/js/v2_bridge.js").read_text(encoding="utf-8")
    assert "cue_text" in (Path(_REPO) / "experiment_game/experiment/trial_v2.py").read_text(encoding="utf-8")
    print("OK operator/bridge wiring")

    orch = (_REPO / "experiment_game/experiment/orchestrator.py").read_text(encoding="utf-8")
    disp = (_REPO / "experiment_game/experiment/ws_dispatch.py").read_text(encoding="utf-8")
    assert 'module_prefix="v3"' in orch and 'module_prefix="v2"' in orch
    assert "_handle_model_eval_grid" in orch and "model_eval_grid" in disp
    assert "v3 微调仅使用 v3" in orch
    assert "subject_models_dir" in orch
    print("OK orchestrator/dispatch")

    # FT filter: only v3_session kept
    from experiment_game.experiment.subject_registry import _read_session_phase_mode

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        v3d = root / "v3_sess"
        v2d = root / "v2_sess"
        v3d.mkdir()
        v2d.mkdir()
        (v3d / "session.meta.json").write_text(
            '{"phase_mode":"v3_session"}', encoding="utf-8"
        )
        (v2d / "session.meta.json").write_text(
            '{"phase_mode":"v2_session"}', encoding="utf-8"
        )
        assert _read_session_phase_mode(v3d) == "v3_session"
        assert _read_session_phase_mode(v2d) == "v2_session"
        kept = [d for d in (v3d, v2d) if _read_session_phase_mode(d) == "v3_session"]
        assert kept == [v3d]
    print("OK v3-only FT filter logic")

    class FakeEvents:
        def emit(self, *a, **k):
            from pylsl import local_clock

            return {"t_lsl": local_clock(), "event": a[0] if a else ""}

    class FakeMarkers:
        def push(self, *a, **k):
            pass

    class FakeBridge:
        def is_paused(self):
            return False

        def broadcast(self, *a, **k):
            pass

    stages: list[str] = []

    def on_stage(stage, ctx, data=None):
        stages.append(stage)

    cfg = V2Config.load_yaml()
    cfg.v3test_rest_s = 0.2
    cfg.v3test_cue_s = 0.1
    cfg.v3test_mi_s = 0.2
    cfg.v3test_judge_interval_s = 0.1
    cfg.v3test_consecutive = 2
    cfg.v3test_n_rest = 2
    cfg.v3test_n_left = 1
    cfg.v3test_n_right = 1
    cfg.iti_s = 0.05
    summary = run_v3test_game(
        FakeEvents(),
        FakeMarkers(),
        FakeBridge(),
        lambda m: None,
        cfg=cfg,
        judgment_fn=None,
        on_stage=on_stage,
        seed=1,
    )
    assert summary["game_mode"] == "v3_test"
    assert summary["score_max"] == 2 * 0.5 + 2 * 1.0
    assert summary["n_trials"] == 4
    assert "game_end" in stages and "cue" in stages and "rest_start" in stages
    print(
        "OK dry-run v3test",
        summary["score"],
        "/",
        summary["score_max"],
        "n_stages",
        len(stages),
    )

    # streak-success dry-run：假判定连续命中 → MI 提前结束 +1；静息连击 +0.5
    hits = {"rest": 0, "mi": 0}

    def judgment_fn(t0, t_rel, ctx):
        lab = int(getattr(ctx, "label", -1))
        phase = getattr(ctx, "score_phase", "mi")
        if phase == "pre_cue_rest":
            hits["rest"] += 1
            return {"pred": 0, "p_max": 0.9, "p_three": [0.9, 0.05, 0.05]}
        hits["mi"] += 1
        return {"pred": lab, "p_max": 0.9, "p_three": [0.05, 0.9, 0.05] if lab == 1 else [0.05, 0.05, 0.9]}

    stages.clear()
    summary2 = run_v3test_game(
        FakeEvents(),
        FakeMarkers(),
        FakeBridge(),
        lambda m: None,
        cfg=cfg,
        judgment_fn=judgment_fn,
        on_stage=on_stage,
        seed=2,
    )
    assert summary2["score"] == summary2["score_max"] == 3.0, summary2
    assert any(t.get("hit") for t in summary2["trials"] if t.get("label") == 0)
    assert all(t.get("hit") for t in summary2["trials"] if t.get("label") in (1, 2))
    print("OK judgment streak scoring", summary2["score"], "hits", hits)

    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    main()
